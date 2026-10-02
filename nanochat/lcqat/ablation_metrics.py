"""
Measurement harness for the LC-QAT ablation claims.

Two claims in the write-up are *empirical*, not structural: that the `asym`
codebook split beats `small`, and that `inv_sqrt_n` codebook gradient scaling
beats `none`. This module is what makes them falsifiable -- it defines the
metrics and the paired protocol, and `scripts/lcqat_ablation.py` drives it.

The protocol is **paired**, and that is the whole point. A single run of each
arm on different random seeds measures seed variance, not the effect. Every
comparison here holds the seed, the data, and the init fixed and varies exactly
one factor, then reports the paired difference. Without that, an apparent win
of a few tenths of a nat is indistinguishable from which seed you happened to
pick.

Honest scope, stated up front:

* `asym` vs `small` can be evaluated here *in isolation* -- it is a static
  property of the codebooks, so a round-trip loss is a fair proxy for the
  claim. That proxy is not the same as end-task perplexity, and the harness
  reports it as what it is.
* `inv_sqrt_n` vs `none` is a property of the *gradient*, not the forward. Its
  direct evidence is the gradient ratio and the resulting parameter movement
  over a fixed number of steps. Convergence over a real run is a stronger claim
  this module does not make on its own.
* DiffusionBlocks noise-range specialization is out of scope here; measuring it
  needs a real d6 training run, which `scripts/lcqat_ablation.py` orchestrates
  and this module only records.

Every metric returns a dataclass rather than a bare float so a leaderboard row
cannot silently mix incompatible numbers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from nanochat.lcqat.linear import GRAD_SCALE_INV_SQRT_N, GRAD_SCALE_NONE


@dataclass(frozen=True)
class ReconstructionResult:
    """Round-trip error of a quantized tensor under one preset.

    Attributes:
        preset: the LC-QAT preset that produced the layers.
        mse: mean squared error between the float and dequantized tensor.
        max_abs_error: largest single-element error.
        signal_power: mean square of the float tensor, i.e. the 0.0 reference
            for a relative measure.
        nmse: `mse / signal_power`, the scale-free error actually compared
            across arms. Raw `mse` is not comparable across presets because the
            tensors differ; `nmse` is.
        used_levels: how many codebook levels are actually hit, out of
            `k_total`. Levels that no element selects are wasted capacity, and
            this is the quantity the `asym` split is really claiming to improve.
    """

    preset: str
    mse: float
    max_abs_error: float
    signal_power: float
    nmse: float
    used_levels: int
    k_total: int
    n_elements: int

    @property
    def level_utilization(self) -> float:
        """Fraction of codebook levels that at least one element selects."""
        return self.used_levels / self.k_total if self.k_total else 0.0


@dataclass(frozen=True)
class PairedComparison:
    """A two-arm comparison under a held-fixed seed.

    `delta` is `variant - baseline`, so a negative `nmse_delta` means the
    variant reconstructs better. Paired because both arms share the seed and
    the data: the difference is attributable to the varied factor rather than
    to initialisation luck.
    """

    name: str
    baseline: str
    variant: str
    nmse_baseline: float
    nmse_variant: float
    nmse_delta: float
    baseline_utilization: float
    variant_utilization: float
    utilization_delta: float
    wins: bool
    seed: int


@dataclass(frozen=True)
class GradScaleObservation:
    """Codebook gradient behaviour under one `grad_scale` setting.

    Attributes:
        grad_scale: `"none"` or `"inv_sqrt_n"`.
        grad_norm: L2 norm of the codebook gradient.
        naive_grad_norm: what the gradient norm would be without the scaling,
            i.e. the `none` arm on the identical graph. Comparing against this
            isolates the scaling from every other source of variation.
        param_delta: L2 distance the codebook parameters moved over a fixed
            number of optimizer steps. This is the number that matters: a
            gradient that is too large does not merely look different, it moves
            the codebook further per step than the same loss justifies.
    """

    grad_scale: str
    grad_norm: float
    naive_grad_norm: float
    grad_ratio: float
    param_delta: float
    steps: int


@dataclass(frozen=True)
class AblationRow:
    """One leaderboard row: a paired comparison plus its provenance.

    The provenance fields exist so a number in the leaderboard can be traced to
    the arm configuration that produced it. A leaderboard of bare floats cannot
    be audited, and an unauditable accuracy number is worth very little.
    """

    experiment: str
    metric: str
    baseline: str
    variant: str
    value_baseline: float
    value_variant: float
    delta: float
    better: str
    seeds: list[int] = field(default_factory=list)
    n_seeds: int = 0
    notes: str = ""

    def as_row(self) -> str:
        """Format as a markdown table row, stable for diffing.

        The trailing newline is load-bearing: the renderer joins rows with
        `"".join(...)`, so a row without one concatenates with the next and the
        whole table collapses onto a single line. That renders as prose rather
        than a table, and it fails silently -- the numbers are all still there,
        just unreadable.
        """
        win = "variant" if self.better == "variant" else "baseline"
        return (
            f"| {self.experiment} | {self.metric} | {self.baseline} | {self.variant} "
            f"| {self.value_baseline:.6g} | {self.value_variant:.6g} "
            f"| {self.delta:+.6g} | {win} | {self.n_seeds} |\n"
        )


def quantization_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> tuple[float, float, float]:
    """Return `(mse, max_abs_error, signal_power)` for a round-trip pair.

    Raises:
        ValueError: on a shape mismatch or a non-finite result. A NaN NMSE would
            otherwise propagate silently into a leaderboard average.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch: original {tuple(original.shape)} vs "
            f"reconstructed {tuple(reconstructed.shape)}"
        )
    # Detached: these are measurement numbers, and a codebook parameter carried
    # into them would both warn and build a graph nobody needs.
    diff = (
        original.detach().to(torch.float32) - reconstructed.detach().to(torch.float32)
    ).abs()
    mse = float(diff.square().mean())
    max_abs = float(diff.max())
    power = float(original.detach().to(torch.float32).square().mean())
    if not (mse >= 0.0 and math.isfinite(mse)):
        raise ValueError(f"non-finite reconstruction error: mse={mse}")
    return mse, max_abs, power


def select_quantizer(layer, which: str):
    """Return one of an `LCQATLinear`'s quantizers by role.

    Args:
        layer: the linear layer.
        which: `"weight"`, `"act"`, or `"out"`.

    Raises:
        ValueError: on an unknown role, or when the layer lacks that quantizer
            (e.g. `"out"` on a layer built with `quantize_out=False`). Falling
            back to the weight quantizer would measure something other than was
            asked, and the two can differ materially.
    """
    names = {
        "weight": "weight_quantizer",
        "act": "act_quantizer",
        "out": "out_quantizer",
    }
    if which not in names:
        raise ValueError(f"which must be one of {sorted(names)}, got {which!r}")
    attr = names[which]
    quantizer = getattr(layer, attr, None)
    if quantizer is None:
        raise ValueError(f"layer has no {attr!r} (which={which!r})")
    return quantizer


def reconstruct_layer(layer, x: torch.Tensor, which: str = "weight") -> torch.Tensor:
    """Run `x` through one of `layer`'s quantizers and back to FP32.

    The round trip an exported artifact actually performs: bucketize into
    indices, then gather the codebook. Dequantized by gathering, never by
    re-running the quantizer's forward, so what is measured is the *artifact's*
    fidelity and not the STE's.

    Args:
        x: the tensor to round-trip -- the weight itself for `"weight"`, or the
            layer's input for the activation/output quantizers.
        which: which quantizer to use; see `select_quantizer`.
    """
    quantizer = select_quantizer(layer, which)
    indices = quantizer.bucketize(x.to(torch.float32))
    codebook = quantizer.get_codebook()
    # Detached deliberately: this measures the exported artifact's fidelity, so
    # the value must not carry the codebook's autograd graph into the metric.
    return codebook.detach()[indices.long()]


def measure_reconstruction(
    layer, x: torch.Tensor, preset: str, which: str = "weight"
) -> ReconstructionResult:
    """Measure round-trip fidelity of one of `layer`'s quantizers.

    `which` selects the quantizer, and it materially changes the result: the
    `asym` preset's one-sided split helps a non-negative `relu^2` activation
    codebook and *hurts* a signed weight codebook. Measuring the wrong one
    yields a correct number that does not test the claim.
    """
    original = x.to(torch.float32)
    reconstructed = reconstruct_layer(layer, original, which)
    mse, max_abs, power = quantization_error(original, reconstructed)

    quantizer = select_quantizer(layer, which)
    indices = quantizer.bucketize(original)
    # "Used" means a level is hit by at least one element. Count distinct
    # values rather than distinct indices so a repeated index is one level.
    used = int(torch.unique(indices).numel())
    k_total = int(quantizer.K)
    return ReconstructionResult(
        preset=preset,
        mse=mse,
        max_abs_error=max_abs,
        signal_power=power,
        nmse=mse / power if power > 0 else float("inf"),
        used_levels=used,
        k_total=k_total,
        n_elements=original.numel(),
    )


def compare_presets(
    name: str,
    baseline: ReconstructionResult,
    variant: ReconstructionResult,
    seed: int,
) -> PairedComparison:
    """Pair two reconstruction measurements under a shared seed."""
    nmse_delta = variant.nmse - baseline.nmse
    return PairedComparison(
        name=name,
        baseline=baseline.preset,
        variant=variant.preset,
        nmse_baseline=baseline.nmse,
        nmse_variant=variant.nmse,
        nmse_delta=nmse_delta,
        baseline_utilization=baseline.level_utilization,
        variant_utilization=variant.level_utilization,
        utilization_delta=variant.level_utilization - baseline.level_utilization,
        # Lower NMSE is better, so the variant wins on a negative delta.
        wins=nmse_delta < 0.0,
        seed=seed,
    )


def aggregate_comparisons(
    name: str, comparisons: list[PairedComparison], metric: str = "nmse"
) -> AblationRow:
    """Average paired comparisons into one leaderboard row.

    Averaging *deltas* rather than the two arms separately is the point of the
    pairing: each seed's difference is computed before the seeds are pooled, so
    per-seed offset cancels instead of inflating the spread.
    """
    if not comparisons:
        raise ValueError(f"no comparisons to aggregate for {name!r}")
    base = sum(c.nmse_baseline for c in comparisons) / len(comparisons)
    var = sum(c.nmse_variant for c in comparisons) / len(comparisons)
    base_util = sum(c.baseline_utilization for c in comparisons) / len(comparisons)
    var_util = sum(c.variant_utilization for c in comparisons) / len(comparisons)
    delta = var - base
    wins = sum(1 for c in comparisons if c.wins)
    return AblationRow(
        experiment=name,
        metric=metric,
        baseline=comparisons[0].baseline,
        variant=comparisons[0].variant,
        value_baseline=base,
        value_variant=var,
        delta=delta,
        better="variant" if wins * 2 > len(comparisons) else "baseline",
        seeds=sorted(c.seed for c in comparisons),
        n_seeds=len(comparisons),
        notes=(
            f"variant wins {wins}/{len(comparisons)} seeds; "
            f"mean level utilization {base_util:.3f} -> {var_util:.3f}"
        ),
    )


def observe_grad_scale(
    layer,
    x: torch.Tensor,
    grad_scale: str,
    steps: int = 5,
    lr: float = 1e-2,
) -> GradScaleObservation:
    """Measure codebook gradient norm and parameter movement under one setting.

    The comparison that makes the claim falsifiable is against `naive_grad_norm`:
    the same graph with the scaling disabled. That isolates the `1/sqrt(N)` factor
    from every other difference between the two code paths.

    `param_delta` is measured over `steps` optimizer steps at a fixed `lr`, so it
    answers the question that actually matters -- does the scaling move the
    codebook further per step than the loss justifies -- rather than merely
    whether the gradients differ.
    """
    if grad_scale not in (GRAD_SCALE_NONE, GRAD_SCALE_INV_SQRT_N):
        raise ValueError(
            f"grad_scale must be one of "
            f"{(GRAD_SCALE_NONE, GRAD_SCALE_INV_SQRT_N)}, got {grad_scale!r}"
        )
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    previous = layer.grad_scale
    # The codebook's own trainable parameters, addressed by name: `get_codebook`
    # returns a *derived* tensor, so stepping that would not be an optimizer
    # step at all.
    target = list(layer.weight_quantizer.parameters())
    if not target:
        raise ValueError("weight_quantizer exposes no trainable parameters")
    before = [p.detach().clone() for p in target]

    layer.grad_scale = grad_scale
    try:
        layer.zero_grad(set_to_none=True)
        out = layer(x.to(torch.float32))
        # A fixed scalar target so the loss is a real signal in both arms; the
        # only thing that differs is how the gradient reaches the codebook.
        loss = out.square().mean()
        loss.backward()
        grad_norm = math.sqrt(
            sum(
                float(p.grad.detach().square().sum())
                for p in target
                if p.grad is not None
            )
        )

        # Same graph, scaling off: the gradient the codebook would have had.
        layer.grad_scale = GRAD_SCALE_NONE
        layer.zero_grad(set_to_none=True)
        out_naive = layer(x.to(torch.float32))
        out_naive.square().mean().backward()
        naive_norm = math.sqrt(
            sum(
                float(p.grad.detach().square().sum())
                for p in target
                if p.grad is not None
            )
        )
        layer.zero_grad(set_to_none=True)

        optimizer = torch.optim.SGD(target, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            layer(x.to(torch.float32)).square().mean().backward()
            optimizer.step()
        after = [p.detach().clone() for p in target]
    finally:
        layer.grad_scale = previous
        layer.zero_grad(set_to_none=True)

    param_delta = math.sqrt(
        sum(float(((a - b) ** 2).sum()) for a, b in zip(after, before, strict=True))
    )
    return GradScaleObservation(
        grad_scale=grad_scale,
        grad_norm=grad_norm,
        naive_grad_norm=naive_norm,
        grad_ratio=grad_norm / naive_norm if naive_norm else float("nan"),
        param_delta=param_delta,
        steps=steps,
    )


def render_leaderboard(rows: list[AblationRow], title: str) -> str:
    """Render leaderboard rows as a markdown table with a provenance header."""
    header = (
        "| experiment | metric | baseline | variant | baseline | variant | "
        "delta | better | seeds |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(row.as_row() for row in rows)
    notes = "\n".join(f"- `{r.experiment}`: {r.notes}" for r in rows if r.notes)
    return f"### {title}\n\n{header}{body}\n**Notes**\n\n{notes}\n"


def row_to_dict(row: AblationRow) -> dict[str, object]:
    """Flatten a row for JSON serialisation (the leaderboard's on-disk form)."""
    return asdict(row)


__all__ = [
    "AblationRow",
    "GradScaleObservation",
    "PairedComparison",
    "ReconstructionResult",
    "aggregate_comparisons",
    "compare_presets",
    "measure_reconstruction",
    "observe_grad_scale",
    "quantization_error",
    "reconstruct_layer",
    "render_leaderboard",
    "row_to_dict",
    "select_quantizer",
]
