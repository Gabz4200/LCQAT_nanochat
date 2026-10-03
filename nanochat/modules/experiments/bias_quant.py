"""The `bias_quant` experiment: is the bias worth one more codebook per layer?

Reports the bias round-trip NMSE on its own, the per-layer parameter and byte
cost of a shared bias table, and whether that table's parameters receive a
gradient and then move.
"""

from __future__ import annotations

import argparse
import math

import torch

from nanochat.models.quant.ablation_metrics import AblationRow, quantization_error
from nanochat.models.quant.linear import LCQATLinear
from nanochat.models.quant.retrofit import PRESETS, CodebookSpec
from nanochat.modules.experiments.common import (
    VARIANT_PRESET,
    count_codebook_cost,
    non_negative_probe,
)
from tests.conftest import build_active_tiny_gpt

#: Marker for the "bias codebook parameters moved" flag in the notes. Checked
#: by string, like the `sparsity` and `act_lut` rows do, so the rendered
#: leaderboard stays readable and the check needs no side channel.
BIAS_CODEBOOK_LIVE = "codebook gradient live: True"
BIAS_CODEBOOK_DEAD = "codebook gradient live: False"


def _weight_kwargs(spec: CodebookSpec) -> dict[str, object]:
    """A `LayerKConfig` cardinality as `from_float`'s weight-codebook kwargs."""
    if isinstance(spec, tuple):
        return {"K_weight_split": spec}
    return {"K_weight": int(spec)}


def _act_kwargs(spec: CodebookSpec) -> dict[str, object]:
    """A `LayerKConfig` cardinality as `from_float`'s activation kwargs."""
    if isinstance(spec, tuple):
        return {"K_act_split": spec}
    return {"K_act": int(spec)}


def bias_probe(preset: str, k_bias: int) -> LCQATLinear:
    """A `c_proj` whose bias goes through a `BiasQuantizer`; the paired baseline.

    nanochat builds every projection with `bias=False`, so the shipped model has
    nothing to quantize on the bias side and a bias codebook cannot be measured
    on it at all. This gives `c_proj` a bias and converts *that one layer*
    in place, so the FP32 baseline and the quantized variant are the same layer
    -- same weight, same probe -- and the only thing that varies is whether the
    bias term is quantized. That is what makes the pair paired.

    The preset's real `down_weight` / `down_act` cardinalities are passed
    through rather than hardcoded, so the weight and activation codebooks under
    the bias are the ones the preset actually ships and the bias error is read
    in the same context the other two terms are measured in. `retrofit_model` is
    deliberately not called: it would re-convert this layer (and `from_float`
    rejects an `LCQATLinear` input), so the conversion is done directly and the
    rest of the model is untouched.

    The layer is rebuilt per seed (see `run_bias_quant`) because the codebook is
    trained below and `build_active_tiny_gpt` is what reseeds the weights.
    """
    model = build_active_tiny_gpt()
    float_proj = model.transformer.h[0].mlp.c_proj
    # `gpt.py` builds every projection with `bias=False`, so `float_proj.bias`
    # is None and there is nothing to give the bias codebook to quantize. A
    # fresh float Linear carrying *this* layer's weight plus a real bias is
    # built here, so the weight and activation codebooks still span the real
    # `c_proj` ranges and only the bias term is new.
    #
    # The bias is drawn at the `std` `build_active_tiny_gpt` itself uses for the
    # zero-initialized projections, off the *global* RNG the caller has already
    # seeded for this sweep -- a dedicated generator seeded to a constant here
    # would hand every seed the identical bias and quietly turn a paired sweep
    # into five copies of one measurement.
    #
    # It must not be left at zero: `BiasQuantizer.init_from_tensor` returns early
    # on an all-identical vector and keeps the codebook on its default span,
    # which for a bias is wider than the data -- the silent, permanent failure
    # its docstring warns about.
    float_with_bias = torch.nn.Linear(
        float_proj.in_features, float_proj.out_features, bias=True
    )
    with torch.no_grad():
        float_with_bias.weight.copy_(float_proj.weight)
        float_with_bias.bias.normal_(mean=0.0, std=0.02)
    cfg = PRESETS[preset]
    replacement = LCQATLinear.from_float(
        float_with_bias,
        grad_scale=cfg.grad_scale,
        quantize_bias=True,
        K_bias=k_bias,
        **_weight_kwargs(cfg.down_weight),
        **_act_kwargs(cfg.down_act),
    )
    model.transformer.h[0].mlp.c_proj = replacement
    return replacement


def run_bias_quant(args: argparse.Namespace) -> list[AblationRow]:
    """Is the bias worth one more codebook per layer? Error, cost, and liveness.

    `dev/HANDOFF_symbiosis.md` §10.2.4 records that nothing in `LCQATLinear`
    quantized the bias, and §10.4-E item 12 asks for exactly this measurement
    before the capability is adopted. Three things decide it, and they are
    reported as three rows rather than folded into one number:

    1. **Fidelity.** The bias round-trip NMSE, quantized vs FP32. Reported on
       its own, never pooled with the weight or activation NMSE: the bias is
       `out_features` numbers against a `D x D` weight matrix, so a pooled
       relative error is ~99.9% weight by element count and would hide whatever
       the bias term does. That is the §9.8 pooled-NMSE mistake, repeated.
    2. **Cost.** `count_codebook_cost` on the quantizer: parameters and artifact
       bytes per layer. Independent of `out_features` -- one shared 1-D table,
       not one per output channel -- which is the only reason it is cheap enough
       to be worth asking about.
    3. **Liveness.** Whether the codebook's own parameters receive a non-zero
       gradient and then move. A codebook that never moves is inert storage, and
       a bias error pinned at exactly 1.0 is its signature: every entry buckets
       to the zero anchor, so the gather is the anchor, so the loss is
       independent of the table.

    The fidelity number is not read off a layer whose bias codebook was left on
    the default `act_init` span. That span (-2..2) is fitted to `relu^2`
    *outputs* and is ~40x wider than a std-0.02 bias, so every entry buckets onto
    the zero anchor, one level is hit, the error is exactly 1.0 and the gradient
    is identically zero -- the silent, permanent failure documented on
    `BiasQuantizer.init_from_tensor`, and measured here rather than assumed.
    `from_float` already performs that fit; the probe calls `init_from_tensor`
    itself because it is the precondition the measurement depends on, not
    something to inherit and hope for.

    Whether the extra table is *worth it* is not decided here and the rows do
    not pretend otherwise: this is a randomly-initialized layer, a handful of
    SGD steps, and a reconstruction error. It measures the price and the
    fidelity cost of the capability, not what it does to a trained model's loss.
    """
    nmse_baseline = 0.0
    nmse_variant = 0.0
    used_levels = 0
    k_total = 0
    moved = 0.0
    ulp_floor = 0.0
    non_zero_grads = 0
    params = 0
    artifact_bytes = 0
    out_features = 0
    in_features = 0

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        layer = bias_probe(VARIANT_PRESET, args.bias_quant_k_bias)
        quantizer = layer.bias_quantizer
        if quantizer is None:
            raise RuntimeError(
                "bias_probe built a layer with no bias_quantizer; the "
                "quantize_bias flag did not reach LCQATLinear.from_float"
            )
        bias = layer.bias.detach().to(torch.float32)
        out_features = int(bias.numel())
        in_features = int(layer.in_features)

        # MANDATORY, and the reason this probe measures anything at all: fit the
        # table to the bias's own range. Without it the span is the layer's
        # `act_init`, which spans relu^2 outputs and therefore overshoots a
        # std-0.02 bias by orders of magnitude -- every entry lands on the zero
        # anchor, the gradient is identically zero, and the error sits at
        # exactly 1.0 forever. `from_float` already did this; doing it again is
        # the probe stating the precondition it depends on rather than trusting
        # a constructor call to have survived.
        quantizer.init_from_tensor(bias)

        # Baseline: the FP32 bias, which round-trips through nothing. Its error
        # is zero by construction, and that is the point of pairing against it
        # -- the question is not "is the quantized bias bad" but "how much does
        # quantizing it cost".
        base_mse, _base_max, base_power = quantization_error(bias, bias.clone())

        indices = quantizer.bucketize(bias)
        reconstructed = quantizer.get_codebook().detach()[indices.long()]
        var_mse, _var_max, var_power = quantization_error(bias, reconstructed)

        nmse_baseline += base_mse / base_power if base_power > 0 else 0.0
        nmse_variant += var_mse / var_power if var_power > 0 else float("inf")
        used_levels += int(torch.unique(indices).numel())
        k_total += int(quantizer.K)

        # Liveness: a real backward pass through the STE, then the codebook's own
        # parameters (never the derived `get_codebook()` tensor, which carries no
        # optimizer state) for `steps` SGD steps. Fixed target, so the loss is a
        # genuine signal in the same way `observe_grad_scale` makes it one.
        # Built once and reused across the SGD steps: `non_negative_probe`
        # seeds its own generator, so every call returned a bit-identical
        # tensor and the loop was allocating one per step to throw away.
        probe = non_negative_probe(layer.in_features, args.n, seed)
        layer.zero_grad(set_to_none=True)
        out = layer(probe)
        out.square().mean().backward()
        target = list(quantizer.parameters())
        if not target:
            raise RuntimeError("bias_quantizer exposes no trainable parameters")
        before = [p.detach().clone() for p in target]
        grad_norm = math.sqrt(
            sum(
                float(p.grad.detach().square().sum())
                for p in target
                if p.grad is not None
            )
        )
        if grad_norm > 0.0:
            non_zero_grads += 1
        optimizer = torch.optim.SGD(target, lr=args.bias_quant_lr)
        for _ in range(args.bias_quant_steps):
            optimizer.zero_grad(set_to_none=True)
            layer(probe).square().mean().backward()
            optimizer.step()
        moved += math.sqrt(
            sum(
                float(((p.detach() - b).square()).sum())
                for p, b in zip(target, before, strict=True)
            )
        )
        # `moved > 0` is the wrong liveness test on its own, and it is exactly
        # the test that made a single-seed sweep report a live codebook as dead.
        # A non-zero gradient does not guarantee a representable update: these
        # latents sit at magnitude ~5, where one fp32 ULP is ~5e-7, so an SGD
        # step of `lr * grad` with `grad` around 1e-5 at the default lr of 1e-2
        # lands at ~1e-7 -- below the spacing between representable values. The
        # parameter is then bit-identical before and after and `moved` is
        # exactly 0.0, which reads as "dead" when the gradient is in fact live
        # and the codebook merely cannot move at that learning rate. Recording
        # the representable floor alongside lets the two cases be told apart.
        ulp_floor += max(
            float(torch.finfo(p.dtype).eps) * float(p.detach().abs().max())
            for p in target
        )
        params, artifact_bytes = count_codebook_cost(quantizer)

    seeds = args.seeds
    nmse_baseline /= seeds
    nmse_variant /= seeds
    avg_moved = moved / seeds
    avg_ulp = ulp_floor / seeds
    # Live means both: a real gradient arrived, AND it moved the table by at
    # least the representable floor. Requiring both is what keeps this check
    # able to fail -- `moved` alone reported a live codebook as dead whenever a
    # single-seed sweep's update landed below one fp32 ULP, while
    # `gradient alone` would report the §9.8 dead-codebook case as live, since
    # that trap has a non-zero *gradient* that is too small to matter for a
    # different reason. Neither half substitutes for the other.
    live = non_zero_grads == seeds and avg_moved >= avg_ulp > 0.0
    live_flag = BIAS_CODEBOOK_LIVE if live else BIAS_CODEBOOK_DEAD
    # `fp32` scores 0 by construction and cannot be beaten, so "better" names
    # the arm with the lower error honestly instead of pretending the quantized
    # arm won. The interesting reading is the *size* of the gap next to the cost.
    better = "variant" if nmse_variant < nmse_baseline else "baseline"
    baseline_label = "fp32 bias"
    variant_label = f"bias_quant@{args.bias_quant_k_bias}"

    return [
        AblationRow(
            experiment="bias_quant_nmse",
            metric="bias_nmse",
            baseline=baseline_label,
            variant=variant_label,
            value_baseline=nmse_baseline,
            value_variant=nmse_variant,
            delta=nmse_variant - nmse_baseline,
            better=better,
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"BIAS NMSE ONLY, reported separately from the weight and "
                f"activation errors on purpose: the bias is {out_features} "
                f"numbers against a {in_features}x{out_features} weight "
                f"matrix, so a pooled relative error would be the weight's by "
                f"element count and would hide the bias term entirely (the 9.8 "
                f"pooled-NMSE mistake). Quantized bias NMSE {nmse_variant:.6g} vs "
                f"{nmse_baseline:.6g} for the FP32 bias, which round-trips through "
                f"nothing and so scores 0 exactly. {used_levels // seeds}/"
                f"{k_total // seeds} levels hit. init_from_tensor() was called "
                "on the bias before this was "
                "measured, because the quantizer's default act_init span "
                "(-2..2, fitted to relu^2 outputs) is ~40x wider than a "
                "std-0.02 bias: every entry then buckets onto the zero anchor, "
                "1 level is hit, the error is exactly 1.0 and the gradient is "
                "identically zero. from_float() performs the same fit, so this "
                "is a precondition the probe states and checks, not a fix it "
                "depends on."
            ),
        ),
        AblationRow(
            experiment="bias_quant_cost",
            metric="bias_codebook_params",
            baseline=baseline_label,
            variant=variant_label,
            value_baseline=0.0,
            value_variant=float(params),
            delta=float(params),
            better="baseline",
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"EXTRA COST PER LAYER: {params} parameters / {artifact_bytes} B "
                f"for one shared K={args.bias_quant_k_bias} table, independent of "
                f"out_features={out_features} (one shared 1-D table, not one per "
                f"output channel). The FP32 baseline carries the bias inline at "
                "no extra parameters, which is why its value is 0 and `better` is "
                "always `baseline` here: this row is a price tag, not a "
                "comparison anyone can win. Judge it against the bias NMSE row."
            ),
        ),
        AblationRow(
            experiment="bias_quant_liveness",
            metric="codebook_param_delta",
            baseline="zero gradient",
            variant=f"codebook gradient live: {live_flag}",
            value_baseline=0.0,
            value_variant=avg_moved,
            delta=avg_moved,
            better="variant" if live else "baseline",
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"{live_flag}. L2 distance the bias codebook's parameters moved "
                f"over {args.bias_quant_steps} SGD steps at lr={args.bias_quant_lr} "
                f"(mean over {seeds} seeds), from a real backward pass through "
                f"LCQATLinear.forward: observed {avg_moved:.6g} against a "
                f"representable floor of {avg_ulp:.6g} (one fp32 ULP at these "
                "latents' magnitude, times the step count). Liveness requires "
                "both a non-zero gradient and movement at or above that floor: "
                "the gradient alone would call the §9.8 dead-codebook case live, "
                "and movement alone calls a codebook that is merely too small to "
                "update at this learning rate dead. A codebook with a live "
                "gradient that does not move at all would mean the learning rate "
                "or the gradient scale is wrong; zero gradient means every entry "
                "bucketizes to the zero anchor and the error is pinned at "
                "exactly 1.0, which is the §9.8 trap this row exists to catch."
            ),
        ),
    ]
