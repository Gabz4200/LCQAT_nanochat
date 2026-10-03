"""
SparsePropLinear module and autograd function for unstructured sparse backpropagation.

Provides a drop-in Linear replacement that uses a static sparsity mask
and routes backward through the AVX2 SparseProp C++ kernels (O(nnz)
instead of O(M*K)).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanochat.models.backbone import Linear
from nanochat.models.quant.linear import (
    apply_trained_activation,
    quantize_with_ste,
)
from nanochat.ops.sparseprop import (
    build_csr_csc_from_mask,
)

#: How a pruning target is distributed across layers.
SCOPE_LAYER = "layer"
SCOPE_GLOBAL = "global"
PRUNE_SCOPES = (SCOPE_LAYER, SCOPE_GLOBAL)

#: SparseProp Sec. 4.1: a module stays on the dense kernel until it reaches
#: this sparsity, then the sparse kernel is chosen by measured forward+backward
#: time. The paper's own number (80%) is the default; the flag exists because
#: the crossover is hardware- and shape-dependent.
DEFAULT_DENSE_THRESHOLD = 0.8


class SparsePropLinearFunction(torch.autograd.Function):
    """Autograd function routing linear backward through the AVX2 kernel.

    Forward: y = SpMM(W * mask, x) + bias
    Backward: gX and gW (at nnz only) via C++ kernel.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, mask):
        """Forward: y = SpMM(W * mask, x) + bias.

        Layout: x [*, in], weight [out, in], y [*, out].
        Kernel uses transposed [in, B] / [out, B] for contiguous batch axis.
        """
        # Flatten batch dims to 2D [B, in] for the kernel
        in_f = x.size(-1)
        out_f, in_f2 = weight.shape
        assert in_f == in_f2, f"dim mismatch x[...,{in_f}] vs W[{out_f},{in_f2}]"
        B = x.numel() // in_f
        x_flat = x.reshape(B, in_f) if x.dim() > 2 else x
        ctx.in_features = in_f
        ctx.out_features = out_f
        ctx.B = B
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape  # save original batch shape for backward reshape
        # Only x and weight are needed by the dense backward below, so the
        # signature carries no CSR/CSC structure: a dense matmul reads none of
        # it. The structure buffers live on the module for the sparse export.
        ctx.save_for_backward(x_flat, weight)
        ctx.mask = mask

        # Dense masked GEMM.
        #
        # This replaced the AVX2 SpMM that walked the CSR nnz list. Same
        # measurement that motivated the dense backward, on the forward at
        # these shapes (out=256, in=1024, batch=2048, sparsity=0.75):
        #
        #     C++ nnz-walking SpMM   80.9 ms
        #     dense masked mm        7.4 ms   (10.9x faster)
        #
        # The SpMM does 4x fewer multiply-accumulates than the GEMM and loses
        # by an order of magnitude, because a per-nnz gather of a B-float row
        # cannot be vectorized the way a packed GEMM micro-kernel is. Counting
        # arithmetic is the wrong optimization target here; achieved FLOPs is.
        #
        # This is a drop-in equivalent, not an approximation: pruned weights
        # hold the *exact* zero anchor (see the note in
        # SparsePropLinearLCQAT.forward), so the dense product over the masked
        # matrix sums the same surviving terms. Measured agreement is ~1e-6
        # relative, i.e. fp32 summation-order noise. Pruned slots stay exactly
        # 0.0 in the weight, which is the contract the sparse export and the
        # mul-less kernels read.
        out = x_flat @ weight.t()  # [B, out_f]

        if bias is not None:
            out = out + bias
        # Restore original batch shape
        if ctx.x_shape[:-1] != out.shape[:-1]:
            out = out.reshape(*ctx.x_shape[:-1], out_f)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        # Flatten to 2D to match forward's flattened layout.
        x_flat, weight = ctx.saved_tensors
        B = ctx.B
        out_f = ctx.out_features

        grad_y_flat = grad_y.reshape(B, out_f)

        # Dense GEMMs, masked rather than iterated over nnz.
        #
        # This replaced a hand-written AVX2 sparse backward that walked the
        # CSR/CSC nnz lists. Measured at the shapes this trains at
        # (out=256, in=1024, batch=2048, sparsity=0.75):
        #
        #     C++ nnz-walking backward   73.9 ms
        #     masked dense GEMM backward  15.6 ms   (4.7x faster)
        #
        # The sparse version does 4x fewer multiply-accumulates (nnz*B vs
        # M*K*B) and still loses by that much, because a per-nnz gather of a
        # B-float row cannot be vectorized the way a blocked GEMM is: each nnz
        # re-walks a B-element row with a stride that defeats the cache, while
        # the GEMM streams both operands once through a packed micro-kernel.
        # Arithmetic count is the wrong thing to optimize on this hardware --
        # achieved FLOPs is. The forward took the same view (see the measurement
        # there): the SpMM does 4x fewer multiply-accumulates and still loses,
        # because a per-nnz gather of a B-float row cannot be vectorized the way
        # a packed GEMM micro-kernel is.
        #
        # Equivalence is exact in structure, not approximate. In the [B, *]
        # layout the tensors are already saved in:
        #
        #   dX_flat[B,in] = gY[B,out] @ W[out,in]
        #   dW[out,in]    = gY[B,out].T @ x[B,in]
        #
        # The sparse walk computed dW only at surviving (m,k) and left the rest
        # at zero; the mask multiply below zeroes precisely those positions, so
        # the two agree to fp32 summation-order noise (~1e-7 relative). dX sums
        # over the already-masked weight, whose pruned entries hold the exact
        # zero anchor, so that product is bit-identical.
        grad_x_flat = grad_y_flat @ weight  # [B, in_f]
        grad_w = grad_y_flat.t() @ x_flat  # [out_f, in_f]

        # Zero grad at masked positions (frozen pruned weights). The multiply
        # is what keeps pruned weights frozen: the GEMM above computes a
        # gradient at every position, including ones the sparse walk skipped.
        grad_w = grad_w * ctx.mask.to(grad_w.dtype)

        # Restore the caller's batch shape.
        grad_x = grad_x_flat.reshape(*ctx.x_shape)

        grad_bias = grad_y.sum(dim=0) if ctx.has_bias else None

        return grad_x, grad_w, grad_bias, None


class SparsePropLinear(Linear):
    """Drop-in Linear replacement with unstructured sparse backpropagation.

    Stores a static boolean mask over the weight matrix and computes
    gradients only at non-zero positions via the AVX2 SparseProp C++
    kernels. Weight entries at masked positions are frozen at zero.

    Args:
        in_features: input dimension
        out_features: output dimension
        sparsity: fraction of weights to prune (0.0-1.0)
        bias: whether to include a bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.75,
        bias: bool = True,
        device=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device)
        self.sparsity = float(sparsity)
        # Static boolean mask [out, in]. persistent=True: the sparsity pattern is
        # trained state, and a non-persistent mask means resume silently re-rolls
        # a random pattern instead of restoring the one the weights were pruned
        # against. The CSR/CSC structures are buffers (not bare attributes) so
        # .to(device) moves them and they reach the checkpoint.
        self.register_buffer(
            "sparsity_mask",
            torch.zeros(out_features, in_features, dtype=torch.bool, device=device),
            persistent=True,
        )
        self.sparsity_mask: torch.Tensor  # type hint for pyright
        self.register_buffer(
            "w_ptr", torch.zeros(1, dtype=torch.int32, device=device), persistent=True
        )
        self.register_buffer(
            "w_col", torch.zeros(0, dtype=torch.int32, device=device), persistent=True
        )
        self.register_buffer(
            "w_ptr_csc",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "w_row", torch.zeros(0, dtype=torch.int32, device=device), persistent=True
        )
        self._init_sparsity()

    def _init_sparsity(self) -> None:
        """Build the initial mask.

        Unconditional, including at `sparsity == 0.0`: an all-False mask buffer
        is not a neutral "no pruning" state, it is an empty CSR structure whose
        forward raises. The dense mask is what `sparsity=0.0` means.
        """
        _init_sparsity_mask(self, self.sparsity)

    def _build_sparse_structure(self):
        """Rebuild CSR/CSC index buffers from current mask."""
        mask = self.sparsity_mask
        w_ptr, w_col, w_ptr_csc, w_row = build_csr_csc_from_mask(mask)
        # Assign through the buffer names so nn.Module keeps the registration
        # (and moves them on .to(device)); a fresh local would shadow the buffer.
        self.w_ptr = w_ptr
        self.w_col = w_col
        self.w_ptr_csc = w_ptr_csc
        self.w_row = w_row

    def _apply_mask(self):
        """Enforce sparsity: zero masked weight entries, rebuild structure."""
        with torch.no_grad():
            self.weight.data[~self.sparsity_mask] = 0.0
        self._build_sparse_structure()

    def _set_mask(self, mask: torch.Tensor) -> None:
        """Update sparsity mask and re-zero inactive weight entries."""
        if mask.shape != self.sparsity_mask.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match the registered "
                f"buffer {tuple(self.sparsity_mask.shape)}"
            )
        self.sparsity_mask.copy_(mask)
        self._apply_mask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: y = SpMM(W * mask, x) + bias."""
        return SparsePropLinearFunction.apply(
            x, self.weight, self.bias, self.sparsity_mask
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, sparsity: float = 0.75
    ) -> "SparsePropLinear":
        """Convert an existing nn.Linear to SparsePropLinear.

        Copies weight/bias and applies a magnitude sparsity mask. All weight
        entries at masked positions are zeroed.
        """
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            sparsity=sparsity,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
            # copy_ into the buffer: a bare attribute assignment would replace the
            # buffer's tensor object and detach it from the module's device/dtype.
            module.sparsity_mask.copy_(magnitude_mask(module.weight, sparsity))
            module._apply_mask()
        return module


class SparsePropLinearLCQAT(SparsePropLinear):
    """SparseProp layer that owns an LCQATLinear's weight/quantizer machinery.

    Unlike SparsePropLinear (which owns its own plain nn.Linear), this class
    *inherits* the weight/quantizer machinery from the LCQATLinear it wraps, so
    the LC-QAT codebook quantizers stay live and trainable. The sparse mask
    is applied on top of the LC-QAT forward value, and sparse backprop walks
    the LC-QAT weight (not a separate dense weight).

    The inner LCQATLinear is *flattened*: its weight/bias/quantizers are
    re-parented into this module's own attributes, so every parameter and
    buffer is registered exactly once. Nesting the LCQATLinear as a submodule
    (``self._inner_lcqat = lcqat_linear``) instead would leave the codebook
    parameters registered twice — once on the wrapper, once on the orphaned
    inner module — producing duplicate optimizer entries and a 2x parameter
    count. The lcqat_linear object is intentionally not retained as an
    attribute (that would re-register it as a submodule); it is left to be
    garbage-collected after its weight/bias/quantizers are re-parented.
    """

    def __init__(self, lcqat_linear, sparsity: float = 0.75):
        from nanochat.models.quant.linear import LCQATLinear

        assert isinstance(lcqat_linear, LCQATLinear), (
            f"Expected LCQATLinear, got {type(lcqat_linear)}"
        )
        # Initialize SparsePropLinear first (sets up mask, ptr/col buffers, etc.)
        super().__init__(
            in_features=lcqat_linear.in_features,
            out_features=lcqat_linear.out_features,
            sparsity=sparsity,
            bias=lcqat_linear.bias is not None,
            device=lcqat_linear.weight.device,
        )
        # Re-parent the LCQATLinear's weight/bias into this module so the
        # codebook quantizers stay live and trainable, and so the module tree
        # registers every parameter exactly once (sharing the Parameter
        # object means nn.Module registers it under this module only).
        self.weight = lcqat_linear.weight
        if lcqat_linear.bias is not None:
            self.bias = lcqat_linear.bias
        # Re-parent EVERYTHING the LCQATLinear owns, not just the two primary
        # quantizers. `out_quantizer` in particular is mandatory: c_q/c_k/c_v and
        # c_fc carry one, and dropping it silently disables the KV-cache
        # quantization path and the fused `quantized_mlp_chain`.
        for attr in (
            "weight_quantizer",
            "act_quantizer",
            "out_quantizer",
        ):
            value = getattr(lcqat_linear, attr, None)
            if value is not None:
                setattr(self, attr, value)
        # Plain-int / str attributes the inference and logging paths read.
        self.K_weight = lcqat_linear.K_weight
        self.K_act = lcqat_linear.K_act
        self.matmul_backend = lcqat_linear.matmul_backend
        self.grad_scale = getattr(lcqat_linear, "grad_scale", "inv_sqrt_n")
        # Export-time buffers (packed_weight_indices / weight_index_format /
        # activation_lut) live on _buffers, not as attributes. Copy them so a
        # re-wrapped module keeps its quantized inference path.
        for buf_name, buf in lcqat_linear._buffers.items():
            if buf is None:
                continue
            self.register_buffer(
                buf_name,
                buf,
                persistent=buf_name not in lcqat_linear._non_persistent_buffers_set,
            )
        # Submodules owned by the LCQATLinear must be re-parented too, not just
        # its buffers. `learnable_activation_lut` (the D9 learned activation
        # table) is registered in `_modules` by attach_learnable_activation_luts,
        # so the buffer copy above misses it and the `NOTE` below discards the
        # rest of the tree: the table's `logits` / `initial_table` parameters
        # were silently dropped the moment this wrapper replaced a `c_fc`, and
        # the checkpoint then saved without them. Loading such a checkpoint
        # failed the strict load with "unexpected key(s)" for every wrapped
        # layer. Copy the whole `_modules` mapping, not a fixed attribute list.
        for mod_name, mod in lcqat_linear._modules.items():
            if mod is not None:
                setattr(self, mod_name, mod)
        # NOTE: do NOT keep a reference to lcqat_linear as an attribute —
        # nn.Module.__setattr__ would register it as a submodule, re-registering
        # its weight/bias/quantizers under a second path and recreating the
        # orphan + duplicate-parameter bug. The re-parented attributes above are
        # the only references needed; lcqat_linear is left to be GC'd.
        #
        # That is exactly why the *engine-owned* layers must not go through this
        # path: `apply_lcqat` replaces the adapter/head Linears in place, so
        # wrapping them here would re-parent a module that is still registered
        # elsewhere in the tree and duplicate every codebook parameter. The
        # engine therefore wraps its own layers via `_wrap_sparseprop`.
        # Apply static sparsity mask (Sparse Transfer) on the shared weight.
        _init_sparsity_mask(self, sparsity)

    def apply_trained_activation(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the re-parented D9 learned table to this layer's output.

        `gpt.py`'s MLP calls this on the layer it is about to run the trained
        table through. The wrapper owns `out_quantizer` and
        `learnable_activation_lut` after re-parenting, so it must expose the
        entry point itself -- otherwise the caller's `getattr` finds nothing,
        and a table that can never be reached never receives a gradient, which
        is the exact failure `LCQATLinear.apply_trained_activation` prevents.
        """
        return apply_trained_activation(
            getattr(self, "learnable_activation_lut", None), self.out_quantizer, y
        )

    def _quantize(self, quantizer, x, numel: int):
        """Quantize `x` through `quantizer` via the LC-QAT STE path.

        SparseProp re-parents the quantizers off the wrapped LCQATLinear but
        does not inherit its methods (the MRO here is SparsePropLinear ->
        nn.Linear, not LCQATLinear), so calling the codebook module directly --
        `quantizer(x)` -- reintroduces two defects the dense layer does not have:

        1. Speed. The codebook's own forward does `codebook[indices]`, a
            differentiable gather, so autograd records an `IndexBackward0`
            whose backward scatters the output gradient back over every weight
            entry with `index_put`. Profiling put that single op at 486 ms --
            over half of the layer's total step, and by far the largest cost in
            the SparseProp path. Routing through `_CodebookSTE` replaces that
            scatter with an explicit `scatter_add_` over the codebook's own
            1-D `indices`, which is far cheaper and is what the dense path does.

        2. Correctness. The direct call also skipped the PRD 2.4 `inv_sqrt_n`
            gradient scale, so a SparseProp layer's codebook received gradients
            `sqrt(numel)` times larger than the same layer un-sparsified.

        Mirrors `LCQATLinear._quantize` by delegating to the same
        `quantize_with_ste` helper, so the two cannot drift apart.
        """
        return quantize_with_ste(quantizer, x, numel, self.grad_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: quantize through codebooks (STE), then sparse matmul.

        The quantized weight flows through the standard autograd graph,
        so the STE in the quantizer routes gW back to codebook params.
        The sparse matmul uses SparsePropLinearFunction for the kernel.
        """
        orig_shape = x.shape
        in_f = x.size(-1)
        x_flat = x.reshape(-1, in_f) if x.dim() > 2 else x

        # Quantize through codebooks (STE — builds autograd graph to codebook params)
        x_q = self._quantize(self.act_quantizer, x_flat, x_flat.numel())
        # No post-dequantization mask multiply: pruned positions already hold an
        # exact 0.0 in the shadow weight, and 0.0 is a codebook *level* (the
        # zero anchor at index m_neg), so it quantizes to exactly 0.0. The old
        # `w_q.value * mask` was numerically identical but discarded the anchor,
        # which cost the codebook any gradient signal from the anchor bin's width
        # and would have made the exported indices disagree with the sparse
        # pattern.
        w_q = self._quantize(self.weight_quantizer, self.weight, self.weight.numel())

        bias = self.bias if self.bias is not None else None
        out = SparsePropLinearFunction.apply(
            x_q.value,
            w_q.value,
            bias,
            self.sparsity_mask,
        )

        if orig_shape[:-1] != out.shape[:-1] or out.shape[-1] != self.out_features:
            out = out.reshape(*orig_shape[:-1], self.out_features)
        # Output quantization, mirroring `LCQATLinear.forward`.
        #
        # This was missing, and that was not a speedup -- it was a silent change
        # to what SparseProp trains. `__init__` re-parents `out_quantizer` (it
        # must, or c_q/c_k/c_v and c_fc lose it and the KV-cache quantization
        # path dies), but the forward never called it. Every SparseProp run was
        # therefore training the 24 layers that carry an out_quantizer with an
        # *unquantized* output, unlike the dense layer, which made the sparse arm
        # look ~30-47% faster end to end because it was doing less work.
        if getattr(self, "out_quantizer", None) is not None:
            out = self._quantize(self.out_quantizer, out, out.numel()).value
        return out

    @classmethod
    def from_lcqat(
        cls, lcqat_linear, sparsity: float = 0.75
    ) -> "SparsePropLinearLCQAT":
        """Wrap an existing LCQATLinear with SparseProp sparse backprop.

        The LCQAT codebook quantizers are preserved and remain trainable.
        """
        return cls(lcqat_linear, sparsity=sparsity)


def apply_static_sparsity_mask(
    model: nn.Module,
    sparsity: float = 0.75,
    target_modules: list[str] | None = None,
    scope: str = SCOPE_LAYER,
) -> float:
    """Initialize sparsity masks on all SparsePropLinear modules.

    `scope="layer"` prunes every matching module to `sparsity` on its own
    (Uniform in SparseProp's terminology). `scope="global"` ranks |W| across all
    matching modules jointly, which the paper measures as the better of the two
    at equal average sparsity.

    Returns the achieved sparsity, measured from the masks rather than assumed:
    a tie at the magnitude threshold can leave a few extra entries alive.
    """
    if scope not in PRUNE_SCOPES:
        raise ValueError(f"scope must be one of {PRUNE_SCOPES}, got {scope!r}")
    modules = [
        module
        for name, module in model.named_modules()
        if isinstance(module, SparsePropLinear)
        and (
            target_modules is None
            or any(name.endswith(suffix) for suffix in target_modules)
        )
    ]
    if scope == SCOPE_GLOBAL:
        return apply_global_pruning(modules, sparsity)
    for module in modules:
        _init_sparsity_mask(module, sparsity)
    if not modules:
        return 0.0
    total = sum(m.weight.numel() for m in modules)
    nnz = sum(int(m.sparsity_mask.sum()) for m in modules)
    return 1.0 - nnz / total if total else 0.0


def collect_sparse_layers(root: nn.Module) -> list[SparsePropLinear]:
    """Every SparsePropLinear reachable from `root`, in module-tree order.

    The gradual-pruning schedule and the global-scope pruner both need the same
    list, and the ordering only has to be deterministic.
    """
    return [m for m in root.modules() if isinstance(m, SparsePropLinear)]


def sparse_layers_above(
    root: nn.Module, dense_threshold: float = DEFAULT_DENSE_THRESHOLD
) -> list[SparsePropLinear]:
    """SparseProp layers whose achieved sparsity reaches `dense_threshold`.

    SparseProp Sec. 4.1 keeps a module on the dense kernel until it is at least
    80% sparse, because below that crossover the sparse kernel's bookkeeping
    costs more than the multiply it saves. This is the measurement-facing half
    of that rule: the report says which layers would actually benefit.
    """
    return [
        m
        for m in collect_sparse_layers(root)
        if m.weight.numel() > 0
        and 1.0 - int(m.sparsity_mask.sum()) / m.weight.numel() >= dense_threshold
    ]


def _init_sparsity_mask(module: SparsePropLinear, sparsity: float) -> None:
    """Magnitude sparsity mask on the shadow weight, with >=1 nnz per row.

    Magnitude, not random (SparseProp Sec. 4.1 "global magnitude pruning
    criterion"). A random mask spends the same budget on a large weight it
    spends on a negligible one, so it is not a pruning criterion at all -- it is
    a way of making nnz/numel exact and leaving accuracy to luck. Pruning the
    smallest |W| keeps the layer's function approximately intact.

    Per-layer here; `apply_global_pruning` re-runs the selection jointly across
    layers when the scope is `global`.
    """
    with torch.no_grad():
        mask = magnitude_mask(module.weight, sparsity)
        module.sparsity_mask.copy_(mask)
        module._apply_mask()


@torch.no_grad()
def magnitude_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask pruning the `sparsity` fraction of smallest |W| entries.

    Returns a mask of the same shape as `weight`; True means "kept".

    The per-row guarantee (>= 1 kept entry) falls out of `topk`: each row keeps
    exactly `max(1, round(n * (1 - sparsity)))` entries, which is >= 1 for every
    sparsity below 1.0. Ties at the threshold are broken by `topk`'s index
    order, so the result is deterministic for a given weight.

    An all-zero row is a special case that a plain threshold gets wrong: every
    entry satisfies `|W| >= 0.0`, so a `>=` comparison would keep the whole row
    and silently report 0.0 sparsity for that layer. nanochat zero-initializes
    `attn.c_proj` and `mlp.c_proj`, so those rows exist in every fresh model.
    Such a row keeps exactly `keep` entries rather than all of them.
    """
    out_f, in_f = weight.shape
    if sparsity <= 0.0:
        return torch.ones(out_f, in_f, dtype=torch.bool, device=weight.device)
    if sparsity >= 1.0:
        # An all-zero layer annihilates the input; every row keeps exactly one
        # entry (the largest) so the layer stays rank-1 rather than dead.
        keep = 1
    else:
        keep = max(1, round(in_f * (1.0 - sparsity)))
    if keep >= in_f:
        return torch.ones(out_f, in_f, dtype=torch.bool, device=weight.device)
    magnitudes = weight.detach().abs().float()
    # topk over the flattened row; -inf at pruned positions keeps a uniform
    # threshold mask on ties instead of an arbitrary subset.
    threshold = magnitudes.topk(keep, dim=1).values[:, -1:].contiguous()
    mask = magnitudes >= threshold
    # Rows whose top-`keep` threshold is exactly 0.0 are all-zero rows, which the
    # `>=` above keeps entirely. Fall back to a topk selection for those, so
    # the per-row keep count is honoured regardless of the weight values.
    degenerate = threshold.squeeze(1) == 0.0
    if bool(degenerate.any()):
        # `topk` on the magnitudes: positions with the largest |W| survive, ties
        # resolved by index order, which is deterministic.
        order = magnitudes.topk(keep, dim=1).indices
        fallback = torch.zeros_like(mask)
        fallback.scatter_(1, order, True)
        mask = torch.where(degenerate.unsqueeze(1), fallback, mask)
    return mask


@torch.no_grad()
def apply_global_pruning(
    modules: list[SparsePropLinear],
    sparsity: float,
    current_sparsity: float = 0.0,
) -> float:
    """Re-prune `modules` jointly to `sparsity` and report the achieved sparsity.

    SparseProp Fig. 6 compares Uniform-GMP against Global-GMP at the same target
    sparsity and finds Global better, so the two scopes are not equivalent and
    neither is the default by construction. Global scope ranks |W| across *all*
    listed modules by one threshold, which lets a layer whose weights are
    uniformly small keep far more of them than a layer with heavy-tailed
    weights -- the per-row cap is applied after the global selection so no row
    is ever emptied.

    `current_sparsity` is the sparsity those modules already carry. Gradual
    pruning raises it toward `sparsity`; a raise never revives a pruned weight,
    which is the monotone behaviour Gradual Magnitude Pruning is defined to
    have.

    Monotonicity is enforced explicitly rather than assumed. Ranking |W| across
    all positions from scratch does *not* produce a nested selection: a pruned
    entry's shadow weight is zeroed by `_apply_mask`, so |W| == 0 and it loses
    every comparison -- but a row whose surviving entries happen to be equal in
    magnitude can still swap one for another as the global threshold moves. The
    current mask is therefore intersected with the fresh selection.
    """
    if not modules:
        return current_sparsity
    if sparsity <= current_sparsity:
        return current_sparsity
    with torch.no_grad():
        # 1. Rank every candidate position jointly and take the global top.
        magnitudes = torch.cat(
            [m.weight.detach().abs().float().reshape(-1) for m in modules]
        )
        target_nnz = int(round(magnitudes.numel() * (1.0 - sparsity)))
        flat_mask = torch.zeros(
            magnitudes.numel(), dtype=torch.bool, device=magnitudes.device
        )
        if target_nnz > 0:
            threshold = magnitudes.topk(target_nnz).values[-1]
            # `>=` may keep more than target_nnz under ties; that is fine (the
            # achieved sparsity is recomputed below and reported truthfully).
            flat_mask = magnitudes >= threshold
        # 2. Intersect with what is already kept (monotonicity), then
        #    re-impose the per-row guarantee on the survivors.
        offset = 0
        for module in modules:
            n = module.weight.numel()
            chunk = flat_mask[offset : offset + n].view_as(module.weight)
            offset += n
            module.sparsity_mask.logical_and_(chunk)
            _give_back_empty_rows(module, module.sparsity_mask)
            module._apply_mask()
    total = sum(m.weight.numel() for m in modules)
    nnz = sum(int(m.sparsity_mask.sum()) for m in modules)
    achieved = 1.0 - nnz / total if total else 0.0
    return achieved


@torch.no_grad()
def _give_back_empty_rows(module: "SparsePropLinear", chunk: torch.Tensor) -> None:
    """Set each all-pruned row's largest-magnitude entry in `chunk` to True.

    Vectorized over rows: a per-row Python loop would be out_f host syncs, which
    is the same cost W3.3 removed from the CSR/CSC builders. Only the empty
    rows are touched (`nonzero` on the row sums), so the common case does no
    host work at all beyond one shape read.
    """
    empty_rows = (chunk.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if empty_rows.numel() == 0:
        return
    magnitudes = module.weight.detach().abs().float()
    best = magnitudes[empty_rows].argmax(dim=1)
    chunk[empty_rows, best] = True


def inject_sparseprop_layers(
    model: nn.Module,
    sparsity: float = 0.75,
    target_modules: list[str] | None = None,
    with_lcqat: bool = False,
) -> nn.Module:
    """Recursively replace linear layers with SparseProp linear layers.

    - If with_lcqat=False (default): replaces nn.Linear (not already
      SparsePropLinear/LCQATLinear) with SparsePropLinear.
    - If with_lcqat=True: also converts LCQATLinear → SparsePropLinearLCQAT,
      preserving the codebook quantizers. SparsePropLinearLCQAT absorbs the
      LCQATLinear's weight/quantizers in-tree (no orphan submodule remains),
      so each LCQATLinear is wrapped exactly once.

    Args:
        model: the model to retrofit
        sparsity: initial sparsity level
        target_modules: optional suffixes e.g. ["qkv", "mlp"]
        with_lcqat: if True, also wrap existing LCQATLinear modules
    """
    from nanochat.models.quant.linear import LCQATLinear

    replacements: list[tuple[str, str, nn.Module]] = []  # (name, kind, module)
    for name, module in model.named_modules():
        kind = None
        if isinstance(module, SparsePropLinear):
            continue
        if with_lcqat:
            if isinstance(module, SparsePropLinearLCQAT):
                continue
            if isinstance(module, LCQATLinear):
                # SparsePropLinearLCQAT absorbs the LCQATLinear's weight/
                # quantizers in-tree and does not retain the LCQATLinear as a
                # submodule, so any LCQATLinear seen here is a genuine one to
                # wrap (no orphan-skip needed).
                kind = "lcqat"
        else:
            if isinstance(module, LCQATLinear):
                continue
            if isinstance(module, nn.Linear):
                kind = "linear"
        if kind is None:
            continue
        if target_modules is not None:
            if not any(name.endswith(suffix) for suffix in target_modules):
                continue
        replacements.append((name, kind, module))

    # Replace bottom-up
    for name, kind, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        if parent_name == "":
            parent = model
        else:
            parent = model.get_submodule(parent_name)
        if parent is None:
            continue

        if kind == "linear":
            assert isinstance(module, nn.Linear), (
                f"Expected nn.Linear, got {type(module)}"
            )
            new_module = SparsePropLinear.from_linear(module, sparsity=sparsity)
        elif kind == "lcqat":
            assert isinstance(module, LCQATLinear), (
                f"Expected LCQATLinear, got {type(module)}"
            )
            new_module = SparsePropLinearLCQAT.from_lcqat(module, sparsity=sparsity)
        else:
            continue
        setattr(parent, child_name, new_module)

    return model


__all__ = [
    "DEFAULT_DENSE_THRESHOLD",
    "PRUNE_SCOPES",
    "SCOPE_GLOBAL",
    "SCOPE_LAYER",
    "SparsePropLinear",
    "SparsePropLinearLCQAT",
    "SparsePropLinearFunction",
    "_give_back_empty_rows",
    "apply_global_pruning",
    "apply_static_sparsity_mask",
    "collect_sparse_layers",
    "inject_sparseprop_layers",
    "magnitude_mask",
    "sparse_layers_above",
    "_init_sparsity_mask",
]
