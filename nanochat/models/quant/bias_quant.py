"""Bias quantization for `LCQATLinear` (opt-in; a measurement until proven).

The shipped LC-QAT path consumes the bias in FP32: `LCQATLinear.forward` passes
`self.bias` straight into `F.linear`, so a layer whose weights *and* activations
both go through learned codebooks still adds one un-quantized term to the sum.
dev/HANDOFF_symbiosis.md §10.2.4 / §10.4-E record this as the one genuine
capability gap against `reference/simple.QuantizedLinear`, which runs the bias
through the same two stages as the weight.

`BiasQuantizer` closes that gap with the *same* primitive as the weight and
activation paths -- `MemoryEfficientLearnedCodebook` -- rather than a second
quantization scheme. That is not tidiness; it is what preserves the two
invariants everything else in this package depends on:

* **The exact structural zero.** The codebook anchors index `m_neg` to FP32
  `0.0` as an index assignment, not a computed value, so `bucketize(0.0)` is
  exactly `m_neg` in any regime. A bias entry that is exactly `0.0` therefore
  dequantizes to exactly `0.0` -- a bias which is structurally absent stays
  structurally absent, instead of becoming a 1e-8 residue added to every row.
* **The dual-gradient STE.** Forward value is exactly `C[Q]`; backward routes
  the identity gradient to the shadow bias *and* the scatter-summed gradient to
  the codebook, so the shadow bias keeps training and the table keeps learning
  from the same signal.

Cost: `K` learned levels per layer, independent of `out_features` (one shared
1-D table, not one per output channel). That is what makes this cheap enough to
be a fair candidate, and it is why it does *not* conflict with
`per_channel_weight` -- that one costs `out_features x K` on the weight side.

Measurement is the point. `measure_bias_quantization` reports the bias NMSE
next to the weight and activation NMSE of the same layer on the same probe, so
"quantize the bias too" is a number in this repo rather than a belief.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nanochat.models.quant.codebook import (
    MemoryEfficientLearnedCodebook,
    QuantizedOutput,
)
from nanochat.models.quant.packing import index_dtype_for_k

if TYPE_CHECKING:
    from nanochat.models.quant.ablation_metrics import ReconstructionResult

#: Default level count for the bias codebook. Matches `K_act`'s shipped default
#: so enabling this on one layer and not another does not change which of the
#: three terms is the coarse one.
DEFAULT_K_BIAS = 15


class BiasQuantizer(nn.Module):
    """One shared 1-D learnable codebook for a layer's bias vector.

    Shapes:
        forward accepts `(out_features,)` -- the layer's bias -- and returns a
        `QuantizedOutput` whose `value` is the dequantized bias (same shape) and
        whose `indices` are the packed alphabet positions export would store.

    Args:
        out_features: length of the bias vector this quantizer serves. Stored so
            a shape mismatch is caught here rather than by a broadcast that
            silently half-works.
        K_bias: total level count for a symmetric codebook.
        K_bias_split: `(m_neg, m_pos)` asymmetric split, which wins over
            `K_bias`. A bias is signed in general but a zero-initialized one is
            one-sided, so the split is what lets either shape be expressed.
        init_min, init_max: initial span of the codebook's two arms. A
            caller-supplied guess is a trap -- see `init_from_tensor`, which is
            required in practice rather than an optimisation.
        grad_scale: `"inv_sqrt_n"` (the shipped default) scales the codebook
            gradient by `1/sqrt(out_features)`; `"none"` uses the plain STE.
            Both quantize the *same* way -- the setting changes only `dL/dC`, so
            switching it can never turn bias quantization itself off.

    Invariant preserved: a bias entry that is exactly `0.0` dequantizes to
    exactly `0.0` (structural zero anchor), and the shadow bias's gradient is
    the identity (STE), so enabling this changes the *forward value* of the
    bias addend and nothing about how the bias learns.
    """

    def __init__(
        self,
        out_features: int,
        K_bias: int = DEFAULT_K_BIAS,
        K_bias_split: tuple[int, int] | None = None,
        init_min: float = -1.0,
        init_max: float = 1.0,
        grad_scale: str = "inv_sqrt_n",
        device=None,
    ):
        super().__init__()
        if isinstance(out_features, bool) or not isinstance(out_features, int):
            raise ValueError(f"out_features must be an int, got {out_features!r}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        # Re-validated rather than trusted: a bad `grad_scale` here would
        # silently pick a branch, which is the one outcome this module exists
        # to rule out. Imported lazily because `linear.py` imports this module.
        from nanochat.models.quant.linear import GRAD_SCALES

        if grad_scale not in GRAD_SCALES:
            raise ValueError(
                f"grad_scale must be one of {GRAD_SCALES}, got {grad_scale!r}"
            )
        self.grad_scale = grad_scale
        self.out_features = out_features
        # `linear.py` imports this module, so the codebook factory is imported
        # lazily for the same reason the grad-scale table is.
        from nanochat.models.quant.linear import _make_codebook

        self.codebook = _make_codebook(
            K_bias, K_bias_split, (init_min, init_max), device=device
        )
        self.m_neg = self.codebook.m_neg
        self.m_pos = self.codebook.m_pos
        self.K = self.codebook.K

    def get_codebook(self) -> torch.Tensor:
        """The FP32 `[K]` level table (shared by every bias entry)."""
        return self.codebook.get_codebook()

    def bucketize(self, x: torch.Tensor) -> torch.Tensor:
        """Codebook index per element, no value gather."""
        self._check_shape(x)
        return self.codebook.bucketize(x)

    def indices(self, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Packed index buffer for `bias`, uint8 for `K <= 255` else int32.

        Exposed for a future export step; nothing in the shipped pipeline packs
        the bias yet, so this is read-only plumbing and not a second code path.
        """
        if bias is None:
            raise ValueError(
                "indices() needs the bias tensor; BiasQuantizer owns the "
                "codebook, not the layer's shadow bias"
            )
        indices = self.bucketize(bias)
        return indices.to(index_dtype_for_k(self.K))

    def forward(self, bias: torch.Tensor) -> QuantizedOutput:
        """Dequantize `bias` through the codebook, with the STE intact.

        Mirrors `LCQATLinear._quantize`, which is the point: both grad scales
        produce the *same* forward value `C[bucketize(bias)]` and differ only
        in the `1/sqrt(N)` factor on `dL/dC`. `N` here is `out_features`, a
        *small* number, so the factor is mild -- unlike the activation path,
        whose `N = B*T*D` is in the millions. It is applied anyway because the
        alternative is a bias codebook whose effective step size silently
        depends on `out_features`.

        Under `grad_scale="none"` the codebook's own forward runs, whose STE
        expression is equivalent apart from the scaling. It is *not* a bypass
        to the raw bias: that would leave the forward un-quantized while the
        other two terms stayed quantized, which measures and trains a different
        model than the flag says.
        """
        self._check_shape(bias)
        # Imported lazily: `linear.py` imports this module, so a module-level
        # import would be circular. The quantizer body itself is shared rather
        # than reimplemented -- a second copy of the dual-gradient expression
        # would be free to drift from the weight path's.
        from nanochat.models.quant.linear import quantize_with_ste

        return quantize_with_ste(self.codebook, bias, bias.numel(), self.grad_scale)

    @torch.no_grad()
    def init_from_tensor(self, x: torch.Tensor, percentile: float = 100.0) -> None:
        """Seed the codebook from `x`'s own range.

        **Required in practice, not an optimisation.** A caller-supplied
        `init_max` is a guess, and when it overshoots the data the failure is
        silent and unrecoverable: every value bucketizes onto the zero anchor,
        the anchor is the only level ever gathered, so the codebook parameters
        receive exactly zero gradient and the table never moves. `per_channel.py`
        measured exactly that (NMSE pinned at 1.0 for 400 steps with
        `init_max=100` on data of magnitude 8), and the bias is the same trap
        with fewer elements to hide in.

        Fits both arms to the data's clipped extremes and rebuilds the step
        parameters from them, so the codebook that comes out spans the data.

        A *degenerate* bias -- all-identical entries, including the all-zero one
        nanochat's `init_weights` leaves behind -- is left on the constructor's
        span instead of being fitted. It has no range to fit, and narrowing to
        the 1e-3 floor would put every level inside the gap between the anchor
        and the constant's own value, so the table stays able to represent the
        values the bias will actually take once it starts training. Every such
        entry lands on the exact zero anchor meanwhile, which is correct: a
        constant bias is constant, and the codebook gets no gradient from it
        until the shadow bias moves.

        Raises:
            RuntimeError: on a meta-device tensor. A meta tensor has no data to
                measure, and fitting from its (meaningless) zeros would install
                a silently wrong span -- `to_empty()` paths hit this before
                `init_weights()`.
        """
        self._check_shape(x)
        if x.is_meta:
            raise RuntimeError(
                "BiasQuantizer.init_from_tensor needs a materialized bias; a "
                "meta tensor has no range to fit. Run to_empty()/init_weights() "
                "first, or construct the codebook with the intended span."
            )
        flat = x.detach().reshape(-1).to(torch.float32)
        if bool((flat == flat[0]).all()):
            return
        lo = torch.quantile(flat, (100.0 - percentile) / 100.0)
        hi = torch.quantile(flat, percentile / 100.0)
        # The floor is on the *span*, for the same reason `per_channel.py` puts
        # it there: clamping the extent to a floor is a maximum on the width, so
        # a near-constant column would still come out near-zero-wide and still
        # duplicate levels. The all-identical case returned above; this catches
        # the merely tiny one.
        span = float(torch.clamp((hi - lo).abs(), min=1e-3))
        fitted = MemoryEfficientLearnedCodebook(
            m_neg=self.codebook.m_neg,
            m_pos=self.codebook.m_pos,
            init_min=-span * 0.5,
            init_max=span * 0.5,
            device=self.codebook.raw_pos_deltas.device
            if self.codebook.raw_pos_deltas is not None
            else self.codebook.compiled_codebook.device,
        )
        # Copy rather than swap: `self.codebook` may already have parameters in
        # an optimizer's param groups (or `del`-ed by compile_for_inference),
        # and replacing the module would orphan them mid-training.
        for name in ("raw_pos_deltas", "raw_neg_deltas"):
            src = getattr(fitted, name, None)
            dst = getattr(self.codebook, name, None)
            if src is None or dst is None:
                continue
            dst.copy_(src)

    def _check_shape(self, x: torch.Tensor) -> None:
        if x.numel() != self.out_features:
            raise ValueError(
                f"bias has {x.numel()} entries, expected out_features="
                f"{self.out_features}; a mismatch would broadcast into a "
                f"silently wrong addend"
            )

    def extra_repr(self) -> str:
        return (
            f"out_features={self.out_features}, m_neg={self.m_neg}, "
            f"m_pos={self.m_pos}, K={self.K}"
        )


@dataclass(frozen=True)
class BiasQuantizationReport:
    """Bias round-trip error next to the weight/activation errors, same layer.

    Kept as three `ReconstructionResult`s rather than one scalar, because the
    question "should the bias be quantized" is only answerable *relative* to
    what the other two terms already cost. A bias NMSE that is small in
    absolute terms can still dominate the layer if the weight NMSE is smaller.

    `activation` is None when the layer has no activation probe tensor to
    measure (the weight and bias terms never need one).
    """

    preset: str
    bias: "ReconstructionResult"
    weight: "ReconstructionResult"
    activation: "ReconstructionResult | None" = None

    @property
    def bias_nmse_over_weight_nmse(self) -> float:
        """Bias NMSE divided by weight NMSE; `inf` when the weight term is 0."""
        if self.weight.nmse == 0.0:
            return float("inf")
        return self.bias.nmse / self.weight.nmse


def _reconstruct(
    quantizer, original: torch.Tensor, preset: str
) -> "ReconstructionResult":
    """Round-trip `original` through `quantizer` and score it.

    Mirrors `ablation_metrics.measure_reconstruction` rather than calling it:
    that function selects a quantizer by a fixed role->attribute table, and
    `bias_quantizer` is not a member of it. The measurement itself is the same
    one -- gather by index, score with `quantization_error`, count used levels
    -- so the three arms of the report are directly comparable.
    """
    # Imported lazily: `ablation_metrics` imports `linear`, which imports this
    # module, so a module-level import would be circular.
    from nanochat.models.quant.ablation_metrics import (
        ReconstructionResult,
        quantization_error,
    )

    codebook = quantizer.get_codebook()
    indices = quantizer.bucketize(original)
    reconstructed = codebook.detach()[indices.long()]
    mse, max_abs, power = quantization_error(original, reconstructed)
    return ReconstructionResult(
        preset=preset,
        mse=mse,
        max_abs_error=max_abs,
        signal_power=power,
        nmse=mse / power if power > 0 else float("inf"),
        used_levels=int(torch.unique(indices).numel()),
        k_total=int(quantizer.K),
        n_elements=int(original.numel()),
    )


def measure_bias_quantization(
    layer, x: torch.Tensor, preset: str
) -> BiasQuantizationReport:
    """Measure the bias round-trip error against the weight and activation.

    Args:
        layer: an `LCQATLinear` built with `quantize_bias=True` and a bias.
        x: the probe input, used for the activation term and as the tensor the
            three measurements are compared *on*. The weight and bias terms
            read the parameters themselves.
        preset: provenance tag copied onto every `ReconstructionResult`.

    Raises:
        ValueError: if the layer has no bias quantizer or no bias. Falling back
            to the weight quantizer would measure something other than was
            asked, and the two differ by orders of magnitude here.
    """
    quantizer = getattr(layer, "bias_quantizer", None)
    if quantizer is None:
        raise ValueError(
            "layer has no bias_quantizer; build it with quantize_bias=True to "
            "measure bias quantization"
        )
    bias = getattr(layer, "bias", None)
    if bias is None:
        raise ValueError("layer has no bias to quantize")

    bias_result = _reconstruct(quantizer, bias.detach().to(torch.float32), preset)
    weight_result = _reconstruct(layer.weight_quantizer, layer.weight.detach(), preset)
    activation_result = _reconstruct(
        layer.act_quantizer, x.detach().to(torch.float32), preset
    )
    return BiasQuantizationReport(
        preset=preset,
        bias=bias_result,
        weight=weight_result,
        activation=activation_result,
    )


__all__ = [
    "DEFAULT_K_BIAS",
    "BiasQuantizer",
    "BiasQuantizationReport",
    "measure_bias_quantization",
]
