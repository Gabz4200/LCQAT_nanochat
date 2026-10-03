"""
Dual-quantized drop-in Linear (LC-QAT PRD sections 3.2 and 6).

Subclasses nanochat's Linear so every existing structural contract keeps
holding (isinstance checks in num_matmul_params / init_weights / fp8 filters,
`.weight` attribute access), while forward executes the PRD dual-quantization
path: quantized input activations x quantized weights, optionally quantizing
the output as well (Q/K/V projections and MLP c_fc, per PRD training spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.backbone import Linear
from nanochat.models.quant.bias_quant import BiasQuantizer
from nanochat.models.quant.codebook import (
    MemoryEfficientLearnedCodebook,
    QuantizedOutput,
    codebook_midpoints,
)
from nanochat.models.quant.packing import index_dtype_for_k
from nanochat.models.quant.per_channel import PerChannelValueCenteredQuantizer

#: PRD 2.4 gradient-scaling modes for the codebook step parameters.
GRAD_SCALE_NONE = "none"
GRAD_SCALE_INV_SQRT_N = "inv_sqrt_n"
GRAD_SCALES = (GRAD_SCALE_NONE, GRAD_SCALE_INV_SQRT_N)


class _CodebookSTE(torch.autograd.Function):
    """Codebook lookup with the PRD 2.3 dual gradient and PRD 2.4 1/sqrt(N) scaling.

    Forward is `codebook[bucketize(x)]`. Backward routes the identity gradient to
    `x` (the STE half, PRD 2.3) and the scatter-summed gradient to the codebook
    (the dual-gradient half), scaled by `1/sqrt(N)`.

    Why 1/sqrt(N): in a 4096x4096 linear, 16.7M elements pool into one K-entry
    codebook. Unscaled, the codebook step parameters receive a sum-reduction
    gradient orders of magnitude larger than the per-weight gradients and
    oscillate or diverge relative to the weights (PRD 2.4).

    `N = numel(x)` is passed in rather than inferred, because it differs per call
    site: the weight path has a fixed `out_features * in_features`, while the
    activation path is dynamic per batch. The PRD specifies `N = numel(X)`.
    """

    @staticmethod
    def forward(ctx, x, codebook, numel: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten for the gather: the codebook is 1-D, so `codebook[indices]`
        # requires a 1-D index regardless of how x is shaped. The flat index is
        # carried on ctx and reshaped back in backward.
        x_flat = x.detach().reshape(-1).to(torch.float32)
        midpoints = codebook_midpoints(codebook)
        # Detach for the index: the assignment is discrete and carries no
        # gradient, exactly as in the plain-STE path.
        indices = torch.bucketize(x_flat, midpoints)
        ctx.save_for_backward(codebook, indices)
        ctx.scale = numel**-0.5
        # The indices are returned as a second output so callers that need them
        # (the fused inference path) reuse this bucketize instead of paying for
        # an identical second one over the same tensor.
        return codebook[indices].reshape(x.shape), indices

    @staticmethod
    def backward(ctx, grad_out, grad_indices):
        codebook, indices = ctx.saved_tensors
        # STE identity: the forward value is C[Q], a step function, so the
        # gradient w.r.t. the continuous input passes through unattenuated.
        grad_x = grad_out
        # Dual gradient: scatter the output gradient onto the assigned levels.
        # `indices` is flat (see forward), so flatten grad_out to match. The
        # sigmoid(softplus) factor from the prefix-sum chain rule comes for free,
        # because autograd differentiates the cumsum of softplus itself.
        # Cast grad to codebook dtype: forward runs in fp32 but downstream
        # grad arrives in activation dtype (bf16 under COMPUTE_DTYPE).
        grad_codebook = (
            torch.zeros_like(codebook)
            .scatter_add_(0, indices, grad_out.reshape(-1).to(codebook.dtype))
            .mul_(ctx.scale)
        )
        return grad_x, grad_codebook, None, None


def quantize_with_ste(
    quantizer: MemoryEfficientLearnedCodebook,
    x: torch.Tensor,
    numel: int,
    grad_scale: str,
) -> QuantizedOutput:
    """Quantize `x` through `quantizer`, honoring the PRD 2.4 gradient scale.

    Under `inv_sqrt_n` the codebook gradient is routed through `_CodebookSTE`,
    which scales it by `1/sqrt(numel)`. Under `none` the codebook's own forward
    is used, whose STE expression is equivalent apart from the scaling -- which
    is exactly the difference being ablated.

    Shared by `LCQATLinear` and `SparsePropLinearLCQAT`: SparseProp re-parents the
    quantizers off the wrapped layer without inheriting its methods, so both
    classes need this body and it must not drift between them.
    """
    if grad_scale != GRAD_SCALE_INV_SQRT_N:
        return quantizer(x)
    codebook = quantizer.get_codebook()
    value, indices = _CodebookSTE.apply(x.to(torch.float32), codebook, numel)
    return QuantizedOutput(
        value=value.to(x.dtype),
        indices=indices.to(index_dtype_for_k(quantizer.K)),
        codebook=codebook,
    )


def lcqat_layer_types() -> tuple[type, ...]:
    """Every module class that *behaves* as an LC-QAT layer, subclass or not.

    `SparsePropLinearLCQAT` re-parents an `LCQATLinear`'s weight, bias and all
    three quantizers into its own attributes instead of inheriting from it --
    nesting the inner layer as a submodule would register every codebook
    parameter twice. The consequence is that it is functionally an LC-QAT layer
    and is not a subclass of one.

    So `isinstance(m, LCQATLinear)` is a question about the *class*, and every
    "is this layer quantized?" call site wants the answer about the layer.
    Measured: with 4 of 12 layers sparse-wrapped, `retrofit_summary` reported 8
    and `strip_lcqat` left 4 quantized layers in a supposedly-float twin.

    Use `is_lcqat_layer`, not this tuple, at call sites.
    """
    from nanochat.models.quant.sparseprop import SparsePropLinearLCQAT

    return (LCQATLinear, SparsePropLinearLCQAT)


def is_lcqat_layer(module: nn.Module) -> bool:
    """Whether `module` is an LC-QAT layer. See `lcqat_layer_types` for why.

    Used by `retrofit_summary`, `prepare_lcqat_before_load`, `strip_lcqat` and
    the export walk, all of which previously asked `isinstance` and so skipped
    every sparse LC-QAT layer.
    """
    return isinstance(module, lcqat_layer_types())


def apply_trained_activation(lut, out_quantizer, y: torch.Tensor) -> torch.Tensor:
    """Map `y` through a *trained* activation table, indexed by the out-codebook.

    The training-side counterpart to `_apply_activation_lut`. Shared by
    `LCQATLinear` and `SparsePropLinearLCQAT`: SparseProp re-parents the
    quantizer off the wrapped layer without inheriting its methods, so both
    classes need this body and the two must not drift.

    The gather goes through the table's own `forward`, not
    `lut.resolved_table()[indices]`. The forward *value* is identical, but the
    table's forward is `soft + hard - soft.detach()` while a manual gather
    returns `hard` alone -- so the manual version silently disabled
    `--lcqat-lut-relaxation` and `--lcqat-act-body` on every SparseProp run and
    the trained table's parameters received no gradient at all. It was also
    the slower of the two, being a cheaper computation than the one it stood
    in for.

    `resolved_table()` stays the right accessor for an *integer* gather (the
    export-time `quantized_mlp_chain`); it is the wrong one when the point is
    to train the table.
    """
    if lut is None:
        raise RuntimeError(
            "apply_trained_activation requires a learnable_activation_lut; "
            "call attach_learnable_activation_luts() first, or use the "
            "float activation."
        )
    if out_quantizer is None:
        raise RuntimeError(
            "apply_trained_activation needs c_fc.out_quantizer: the "
            "trained table is indexed by this layer's *output* codebook."
        )
    with torch.no_grad():
        indices = out_quantizer.bucketize(y).reshape(-1)
    return lut(indices).reshape(y.shape).to(dtype=y.dtype)


def _make_codebook(
    k: int,
    split: tuple[int, int] | None,
    init: tuple[float, float],
    device=None,
) -> MemoryEfficientLearnedCodebook:
    """Build a codebook from either a symmetric `k` or an asymmetric `split`.

    `split` is `(m_neg, m_pos)` and wins when given; otherwise `k` is split
    symmetrically, which is what the pre-asymmetric callers pass.
    """
    if split is None:
        return MemoryEfficientLearnedCodebook.from_k(
            k, init_min=init[0], init_max=init[1], device=device
        )
    m_neg, m_pos = (int(split[0]), int(split[1]))
    return MemoryEfficientLearnedCodebook(
        m_neg=m_neg, m_pos=m_pos, init_min=init[0], init_max=init[1], device=device
    )


class LCQATLinear(Linear):
    """Drop-in replacement for Linear with configurable odd codebooks.

    Weight and activation quantizers are always present; `quantize_out` adds a
    third codebook on the output (PRD: training must quantize the output of the
    Q/K/V final projections; the MLP c_fc output quantizer is the input side of
    the fused activation LUT, PRD 6).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        K_weight: int = 3,
        K_act: int = 15,
        quantize_out: bool = False,
        out_k: int | None = None,
        weight_init: tuple[float, float] = (-0.1, 0.1),
        act_init: tuple[float, float] = (-2.0, 2.0),
        device=None,
        dtype=None,
        K_weight_split: tuple[int, int] | None = None,
        K_act_split: tuple[int, int] | None = None,
        out_split: tuple[int, int] | None = None,
        grad_scale: str = GRAD_SCALE_INV_SQRT_N,
        per_channel_weight: bool = False,
        quantize_bias: bool = False,
        K_bias: int | None = None,
        K_bias_split: tuple[int, int] | None = None,
    ):
        """
                `K_weight` / `K_act` / `out_k` give a symmetric split, which is the
                pre-asymmetric API and stays supported. Pass `K_weight_split`,
                `K_act_split` or `out_split` as `(m_neg, m_pos)` to request an
                asymmetric codebook instead; a split takes precedence over the matching
                `K` for that codebook. `m_neg = 0` yields a one-sided (non-negative)
                codebook, which is the right shape for the `relu(x).square()` MLP hidden
                tensor: a symmetric codebook would put half its levels on a sign the
                tensor never takes.

                `grad_scale` selects the PRD 2.4 codebook gradient scaling:
                "inv_sqrt_n" (default) scales the codebook gradient by `1/sqrt(N)`, and
                "none" uses the plain STE. Exposed as a switch because the two are not
                equally good by construction -- with `N = B*T*D` in the millions the
                scaled act-codebook gradient is ~1000x smaller, so it interacts
                multiplicatively with `--codebook-lr` and has to be measured.

                `per_channel_weight` replaces the *weight* quantizer with one table per
                output channel (PRD 3.4). It is off by default and off for the
                activation, for two reasons. The cost is `out_features x K` levels
                rather than `K`, which is a table-size change and not merely a rounding
                change. And the fused inference path consumes a single shared table, so
        enable this makes the layer un-exportable -- `_apply_activation_lut`
                and the packed-index kernels read `weight_quantizer.get_codebook()` as a
                1-D tensor and would silently misread a 2-D one.

                `quantize_bias` routes the bias through one more shared learned
                codebook (`BiasQuantizer`), so a layer's three addends -- quantized
                activation, quantized weight, and the bias -- are all quantized. It is
                off by default and off for the shipped model, because `gpt.py` builds
                every projection with `bias=False`: nothing in the current GPT consumes
                it, so enabling this by default would add a codebook per layer that no
                forward pass ever reads. `K_bias` / `K_bias_split` size that codebook
                (`K_bias` defaults to `K_act`); a split wins over `K_bias`, matching the
                other three codebooks.
        """
        if grad_scale not in GRAD_SCALES:
            raise ValueError(
                f"grad_scale must be one of {GRAD_SCALES}, got {grad_scale!r}"
            )
        if quantize_bias and not bias:
            raise ValueError(
                "quantize_bias=True requires bias=True: there is nothing to "
                "quantize on a bias-free layer, and building the codebook anyway "
                "would add parameters no forward pass reads."
            )
        if quantize_bias and (K_bias_split is not None) and K_bias is not None:
            raise ValueError(
                f"pass K_bias={K_bias} or K_bias_split={K_bias_split}, not both; "
                f"a split and a cardinality describe the same codebook two ways."
            )
        self.grad_scale = grad_scale
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.weight_quantizer = _make_codebook(
            k=K_weight, split=K_weight_split, init=weight_init, device=device
        )
        self.act_quantizer = _make_codebook(
            k=K_act, split=K_act_split, init=act_init, device=device
        )
        out_quantizer = (
            _make_codebook(
                k=self.act_quantizer.K if out_k is None else out_k,
                split=out_split,
                init=act_init,
                device=device,
            )
            if quantize_out
            else None
        )
        # Assign the out quantizer last: it is built from the act quantizer's K,
        # so K_act must already be readable when this runs.
        self.out_quantizer = out_quantizer
        # Per-channel weight table (PRD 3.4), opt-in. Stored alongside rather
        # than replacing `weight_quantizer`: the shared table stays readable so
        # K_width, the export planner, and the optimizer partition keep working,
        # and `weight_quantizer` becomes the fallback whenever the fused path
        # needs a 1-D table.
        self.per_channel_weight_quantizer = (
            PerChannelValueCenteredQuantizer(
                num_channels=out_features,
                m_neg=K_weight_split[0] if K_weight_split else (K_weight - 1) // 2,
                m_pos=K_weight_split[1] if K_weight_split else K_weight // 2,
            )
            if per_channel_weight
            else None
        )
        # Bias codebook (opt-in). A single shared [K] table over the bias
        # vector, re-fitted to the bias itself immediately below.
        #
        # It does NOT conflict with `per_channel_weight`: that one is
        # `out_features x K` on the *weight* side, this is `K` on the bias side,
        # and neither shadows the other's table.
        self.bias_quantizer = (
            BiasQuantizer(
                out_features=out_features,
                K_bias=K_act if K_bias is None else K_bias,
                K_bias_split=K_bias_split,
                init_min=act_init[0],
                init_max=act_init[1],
                grad_scale=self.grad_scale,
                device=device,
            )
            if quantize_bias
            else None
        )
        if self.bias_quantizer is not None:
            # Fit to the freshly-initialized bias, not to the `act_init` span
            # above, for the reason `BiasQuantizer.init_from_tensor` documents:
            # a span that overshoots the data is silent and permanent -- every
            # entry bucketizes onto the zero anchor, so the codebook receives no
            # gradient at all. Measured here: on a direct-constructed layer the
            # unscaled codebook gradients were exactly 0.0. `from_float` refits
            # after copying the float bias in.
            self.bias_quantizer.init_from_tensor(self.bias.detach())
        # Total level counts, kept as plain ints: packing, the KV cache, and the
        # index kernels all key off K, not off the split.
        self.K_weight = self.weight_quantizer.K
        self.K_act = self.act_quantizer.K
        # Backend for the quantized-inference index path (active only once
        # export/load has installed packed_weight_indices buffers).
        self.matmul_backend = "cpu"

    def _quantize(self, quantizer: MemoryEfficientLearnedCodebook, x, numel: int):
        """Quantize `x` through `quantizer`, honoring the PRD 2.4 gradient scale."""
        return quantize_with_ste(quantizer, x, numel, self.grad_scale)

    def _quantize_conditioned(self, quantizer, x, sigma):
        """Quantize through a sigma-conditioned codebook, with PRD 2.4 scaling.

        `numel` is the activation count, matching the static path, so switching
        `--db-sigma-codebook` on does not silently change the codebook step size.

        The scale is applied *inside* the codebook, on the dequantized gather
        only. Applying it to the returned value instead would also scale the
        STE identity term, shrinking the gradient reaching the input activation
        by `1/sqrt(N)` as well -- a silent change to how the model trains, and
        one the static `_CodebookSTE` path deliberately avoids.
        """
        scale = x.numel() ** -0.5 if self.grad_scale == GRAD_SCALE_INV_SQRT_N else 1.0
        return quantizer(x, sigma, scale=scale)

    def _dequantized_bias(self, dtype: torch.dtype) -> torch.Tensor | None:
        """The bias addend, quantized when `quantize_bias` is on.

        Returns None for a bias-free layer, and returns the *raw FP32* bias
        (cast to `dtype`) when `bias_quantizer` is None -- which is the shipped
        path, byte-for-byte.
        """
        if self.bias is None:
            return None
        if self.bias_quantizer is None:
            return self.bias.to(dtype=dtype)
        # `grad_scale` is a plain attribute that existing callers mutate after
        # construction (`test_w2_grad_scale` does exactly that), so the
        # quantizer's copy can go stale. Synced here rather than by a property
        # because `grad_scale` is assigned before `super().__init__()`, and a
        # property setter would then fire on a not-yet-initialized module.
        self.bias_quantizer.grad_scale = self.grad_scale
        return self.bias_quantizer(self.bias).value.to(dtype=dtype)

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Quantize and matmul.

        `sigma` is the per-batch-element noise level, shaped `(B, 1, 1)`. It is
        required when -- and only when -- a quantizer is sigma-conditioned, and
        raises otherwise rather than silently falling back to an unconditioned
        codebook: a caller that believes it is training a conditioned model and
        is not would get numbers that silently mean something else.

        The batch dimension of `x` is what the conditioning is resolved against,
        so `sigma.numel() == x.shape[0]` is the contract. The weight
        quantizer is never conditioned -- one weight matrix feeds every noise
        level, so conditioning it would need a separate matrix per sigma.
        """
        if getattr(self.act_quantizer, "needs_sigma", False):
            if sigma is None:
                raise ValueError(
                    "this layer's activation quantizer is sigma-conditioned, so "
                    "forward() requires the per-batch noise level. Pass "
                    "sigma=(B, 1, 1) or disable --db-sigma-codebook."
                )
            x_q = self._quantize_conditioned(self.act_quantizer, x, sigma)
        else:
            x_q = self._quantize(self.act_quantizer, x, x.numel())
        if "packed_weight_indices" in dict(self.named_buffers()):
            if self.bias_quantizer is not None:
                raise ValueError(
                    "packed_weight_indices are installed, so this layer is on "
                    "the fused inference path, which has no bias-index buffer "
                    "to read. Export does not pack bias indices yet, so a "
                    "quantize_bias layer would silently add its raw shadow "
                    "bias -- different from what it was trained with. Train "
                    "without quantize_bias, or strip the packed buffers."
                )
            if self.per_channel_weight_quantizer is not None:
                raise ValueError(
                    "packed_weight_indices are installed, so this layer is on the "
                    "fused inference path, which reads weight_quantizer's "
                    "single shared [K] table. A per-channel [C, K] table cannot "
                    "be expressed as packed indices. Re-export without "
                    "per_channel_weight, or drop the packed buffers."
                )
            return self._forward_quantized(x, x_q)
        if self.per_channel_weight_quantizer is not None:
            # Per-channel weights: one `[C, K]` table, gathered row-wise. No
            # `1/sqrt(N)` codebook scale here -- the per-channel table has
            # `C x K` parameters rather than `K`, so the same constant would not
            # mean the same thing it does for a shared table, and applying a
            # scale derived from the shared path would silently change how this
            # variant trains relative to the one it is being compared against.
            w_q = self.per_channel_weight_quantizer(self.weight).value
            bias = self._dequantized_bias(x.dtype)
            y = F.linear(x_q.value, w_q.to(dtype=x.dtype), bias)
            if self.out_quantizer is not None:
                y = self._quantize(self.out_quantizer, y, y.numel()).value
            return y
        # Fixed N: one weight matrix feeds one codebook, so the scale is constant
        # across steps (unlike the activation path, whose N varies per batch).
        w_q = self._quantize(self.weight_quantizer, self.weight, self.weight.numel())
        bias = self._dequantized_bias(x.dtype)
        y = F.linear(x_q.value, w_q.value.to(dtype=x.dtype), bias)
        if self.out_quantizer is not None:
            y = self._quantize(self.out_quantizer, y, y.numel()).value
        return y

    def _forward_quantized(self, x: torch.Tensor, x_q: QuantizedOutput) -> torch.Tensor:
        """Inference path: fetch weights through packed IDs + LUT (no fp32 weight).

        Activations quantize to IDs as usual; the matmul resolves both sides
        through their codebooks at fetch time. Requires the buffers installed
        by export_lcqat_checkpoint (packed_weight_indices + weight_index_format).
        """
        from nanochat.ops import dispatch_index_linear

        if self.bias_quantizer is not None:
            raise ValueError(
                "packed path has no bias-index buffer; train without "
                "quantize_bias or strip packed buffers."
            )

        leading = x.shape[:-1]
        act_ids = x_q.indices.reshape(-1, self.in_features)
        y = dispatch_index_linear(
            act_ids,
            x_q.codebook,
            self.packed_weight_indices,
            self.weight_quantizer.get_codebook(),
            self.in_features,
            int(self.weight_index_format),
            backend=self.matmul_backend,
        )
        y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        if self.out_quantizer is not None:
            y = self.out_quantizer(y).value
        return y

    def quantized_mlp_chain(
        self,
        x: torch.Tensor,
        next_linear: "LCQATLinear",
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused quantized MLP chain (PRD 6): c_fc index matmul -> out IDs
        -> relu^2 `activation_lut` table -> c_proj index matmul fed with
        pre-quantized IDs. Float math is only the two matmul accumulations;
        the elementwise op is an index gather.

        Self is c_fc, `next_linear` is c_proj. Requires the buffers
        installed by export (packed weights + activation_lut on self).
        """
        from nanochat.ops import dispatch_index_linear

        if self.out_quantizer is None:
            raise RuntimeError("fused MLP chain requires c_fc.out_quantizer")
        if "activation_lut" not in dict(self.named_buffers()):
            raise RuntimeError("fused MLP chain requires the activation_lut buffer")
        if "packed_weight_indices" not in dict(next_linear.named_buffers()):
            raise RuntimeError("fused MLP chain requires packed c_proj weights")
        if self.activation_lut.dtype != torch.uint8:
            raise ValueError(
                f"fused MLP chain supports K_act <= 255 (uint8 table), got "
                f"{self.activation_lut.dtype}"
            )
        if getattr(getattr(self, "act_quantizer", None), "needs_sigma", False):
            if sigma is None:
                raise ValueError(
                    "quantized_mlp_chain requires sigma for sigma-conditioned "
                    "activation codebooks"
                )

        leading = x.shape[:-1]
        if getattr(getattr(self, "act_quantizer", None), "needs_sigma", False):
            x_q = self.act_quantizer(x, sigma=sigma)  # type: ignore[call-arg]
        else:
            x_q = self.act_quantizer(x)
        y = dispatch_index_linear(
            x_q.indices.reshape(-1, self.in_features),
            x_q.codebook,
            self.packed_weight_indices,
            self.weight_quantizer.get_codebook(),
            self.in_features,
            int(self.weight_index_format),
            backend=self.matmul_backend,
        )
        y_ids = self.out_quantizer(y).indices.reshape(-1, self.out_features)
        z_ids = self._apply_activation_lut(y_ids)
        out = dispatch_index_linear(
            z_ids.reshape(-1, next_linear.in_features),
            next_linear.act_quantizer.get_codebook(),
            next_linear.packed_weight_indices,
            next_linear.weight_quantizer.get_codebook(),
            next_linear.in_features,
            int(next_linear.weight_index_format),
            backend=self.matmul_backend,
        )
        out = out.reshape(*leading, next_linear.out_features)
        if next_linear.bias is not None:
            out = out + next_linear.bias.to(dtype=out.dtype)
        return out

    def apply_trained_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply a *trained* activation table to this layer's float output.

        The training-side counterpart to `_apply_activation_lut`. Without it a
        `learnable_activation_lut` is reachable only from
        `quantized_mlp_chain`, which export calls -- so its parameters would
        never receive a gradient and `--lcqat-lut-relaxation` /
        `--lcqat-act-body` would be flags that change nothing during training.

        Returns the FP32 post-activation value that the downstream layer's act
        quantizer will consume. `LearnableIndexLut.forward` returns exactly the
        value its `resolved_table()` gathers, so this path and the fused chain
        emit the same number; only the gradient differs.

        Falls back to the caller's own activation when no table is attached, so
        the float path is unchanged by default.
        """
        return apply_trained_activation(
            getattr(self, "learnable_activation_lut", None), self.out_quantizer, y
        )

    def _apply_activation_lut(self, y_ids: torch.Tensor) -> torch.Tensor:
        """Map c_fc output indices through the activation table.

        Reads `learnable_activation_lut` when one is attached, so a trained
        table is what the fused chain actually gathers with -- the alternative
        is a training path and an inference path that quietly use different
        activation functions, which would show up only as an accuracy gap.

        Returns an index tensor whose dtype matches the frozen table's (uint8 for
        K_act <= 255), because the downstream `dispatch_index_linear` contract
        requires uint8 act indices. A learnable table's `resolved_table()` is
        int64, so returning it unconverted would break the fused chain.
        """
        learnable = getattr(self, "learnable_activation_lut", None)
        if learnable is not None:
            table = learnable.resolved_table().to(y_ids.device)
            return table[y_ids.long()].to(self.activation_lut.dtype)
        return self.activation_lut[y_ids.long()]

    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        K_weight: int = 3,
        K_act: int = 15,
        quantize_out: bool = False,
        out_k: int | None = None,
        act_init: tuple[float, float] = (-2.0, 2.0),
        K_weight_split: tuple[int, int] | None = None,
        K_act_split: tuple[int, int] | None = None,
        out_split: tuple[int, int] | None = None,
        grad_scale: str = GRAD_SCALE_INV_SQRT_N,
        per_channel_weight: bool = False,
        quantize_bias: bool = False,
        K_bias: int | None = None,
        K_bias_split: tuple[int, int] | None = None,
    ) -> "LCQATLinear":
        if is_lcqat_layer(mod):
            raise ValueError(
                "from_float expects a plain float Linear, got an LCQATLinear"
            )
        if mod.weight.is_meta:
            raise RuntimeError(
                "LCQATLinear.from_float requires materialized weights; "
                "run to_empty()/init_weights() before retrofitting"
            )
        # Weight quantizer spans the observed weight range; a zero span (zero-init
        # projections) is floored inside the codebook so the zero anchor stays exact.
        w_max = mod.weight.detach().abs().max().item()
        # `and mod.bias is not None` rather than plain `quantize_bias`: `gpt.py`
        # builds every projection with `bias=False`, so a config-level flag
        # must degrade to "nothing to do" on a bias-free layer instead of
        # raising on the first one it reaches. Direct construction
        # (`LCQATLinear(..., quantize_bias=True, bias=False)`) still raises,
        # because there the caller has both flags in hand.
        quantize_bias = bool(quantize_bias and mod.bias is not None)
        new_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=(mod.bias is not None),
            K_weight=K_weight,
            K_act=K_act,
            quantize_out=quantize_out,
            out_k=out_k,
            weight_init=(-w_max, w_max),
            act_init=act_init,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            K_weight_split=K_weight_split,
            K_act_split=K_act_split,
            out_split=out_split,
            grad_scale=grad_scale,
            # Forwarded verbatim: `retrofit_model` passes it through, and
            # dropping it here breaks every LC-QAT retrofit at model-build time
            # on an unexpected keyword, with no per-channel flag even set.
            # `pyrefly` catches a dropped keyword statically.
            per_channel_weight=per_channel_weight,
            quantize_bias=quantize_bias,
            K_bias=K_bias,
            K_bias_split=K_bias_split,
        )
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        if new_mod.bias_quantizer is not None:
            # Fit to the *copied* bias, not the ctor's act_init span. Same
            # reason as the `w_max` measurement above: a span that overshoots the
            # data is silent and permanent, because every entry then lands on
            # the zero anchor and the codebook receives no gradient at all.
            new_mod.bias_quantizer.init_from_tensor(new_mod.bias.detach())
        return new_mod
