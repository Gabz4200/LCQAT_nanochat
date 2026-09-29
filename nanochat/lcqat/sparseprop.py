"""
SparsePropLinear module and autograd function for unstructured sparse backpropagation.

Provides a drop-in Linear replacement that uses a static sparsity mask
and routes backward through the AVX2 SparseProp C++ kernels (O(nnz)
instead of O(M*K)).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanochat.gpt import Linear
from nanochat.lcqat.ops.sparseprop import (
    _csc_col_indices,
    _nnz_row_indices,
    build_csr_csc_from_mask,
    gather_values_from_dense,
    sparseprop_backward_cpu,
    sparseprop_forward_cpu,
)


class SparsePropLinearFunction(torch.autograd.Function):
    """Autograd function routing linear backward through the AVX2 kernel.

    Forward: y = SpMM(W * mask, x) + bias
    Backward: gX and gW (at nnz only) via C++ kernel.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, mask, w_ptr, w_col, w_ptr_csc, w_row):
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
        ctx.x_shape = x.shape  # save original shape for backward reshape

        ctx.save_for_backward(x_flat, weight)
        ctx.mask = mask
        ctx.w_ptr = w_ptr
        ctx.w_col = w_col
        ctx.w_ptr_csc = w_ptr_csc
        ctx.w_row = w_row

        # Transpose to [in, B] for kernel (contiguous batch axis)
        x_t = x_flat.t().contiguous()  # [in_f, B]
        # Gather sparse weight values (CSR order)
        w_val = gather_values_from_dense(weight, w_col, w_ptr, out_f)

        y_t = sparseprop_forward_cpu(x_t, w_val, w_col, w_ptr, bias, out_f)
        y_flat = y_t.t().contiguous()  # [B, out_f]
        # Restore original batch shape
        if ctx.x_shape != y_flat.shape:
            out = y_flat.reshape(*ctx.x_shape[:-1], out_f)
        else:
            out = y_flat
        return out

    @staticmethod
    def backward(ctx, grad_y):
        # Transpose grad_y to [out, B] and x to [in, B] for kernel
        # Flatten to 2D to match forward's flattened layout
        x_flat, weight = ctx.saved_tensors
        B = ctx.B
        in_f = ctx.in_features
        out_f = ctx.out_features
        nnz = ctx.w_col.numel()

        grad_y_flat = grad_y.reshape(B, out_f)
        grad_y_t = grad_y_flat.t().contiguous()  # [out_f, B]
        x_t = x_flat.t().contiguous()  # [in_f, B]

        # Gather sparse weight values (CSR order for dW, CSC order for dX)
        w_val = gather_values_from_dense(weight, ctx.w_col, ctx.w_ptr, out_f)
        # For dX (CSC): weight[m,k] where m=w_row[p], k=col_idx_csc[p]
        col_idx_csc = _csc_col_indices(ctx.w_ptr_csc, in_f, nnz, grad_y.device)
        lin_idx_csc = ctx.w_row.to(weight.device) * in_f + col_idx_csc.to(weight.device)
        w_val_csc = weight.view(-1)[lin_idx_csc]

        # Sparse backward via C++ kernel
        gX_t, gW_val = sparseprop_backward_cpu(
            grad_y_t,
            x_t,
            w_val,
            ctx.w_col,
            ctx.w_ptr,
            w_val_csc,
            ctx.w_row,
            ctx.w_ptr_csc,
            out_f,
            in_f,
        )

        # Transpose gradients back to standard layout
        gX_flat = gX_t.t().contiguous()  # [B, in_f]
        # Restore original input batch shape
        if ctx.x_shape != gX_flat.shape:
            gX = gX_flat.reshape(*ctx.x_shape)
        else:
            gX = gX_flat

        # Scatter gW_val into dense [out, in]
        gW = torch.zeros(out_f, in_f, dtype=grad_y.dtype, device=grad_y.device)
        row_idx = _nnz_row_indices(ctx.w_ptr, out_f)
        lin_idx = row_idx.to(weight.device) * in_f + ctx.w_col.to(weight.device)
        gW.view(-1)[lin_idx] = gW_val

        gBias = grad_y.sum(dim=0) if ctx.has_bias else None

        # Zero grad at masked positions (frozen pruned weights)
        gW = gW * ctx.mask.to(gW.dtype)

        return gX, gW, gBias, None, None, None, None, None


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
        # Static boolean mask [out, in]
        self.register_buffer(
            "sparsity_mask",
            torch.zeros(out_features, in_features, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.sparsity_mask: torch.Tensor  # type hint for pyright
        self.w_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        self.w_col = torch.zeros(0, dtype=torch.int32, device=device)
        self.w_ptr_csc = torch.zeros(1, dtype=torch.int32, device=device)
        self.w_row = torch.zeros(0, dtype=torch.int32, device=device)
        self._init_sparsity()

    def _init_sparsity(self) -> None:
        """Initialize sparsity mask if sparsity > 0."""
        if self.sparsity > 0.0:
            _init_sparsity_mask(self, self.sparsity)

    def _build_sparse_structure(self):
        """Rebuild CSR/CSC index buffers from current mask."""
        mask = self.sparsity_mask
        self.w_ptr, self.w_col, self.w_ptr_csc, self.w_row = build_csr_csc_from_mask(
            mask
        )

    def _apply_mask(self):
        """Enforce sparsity: zero masked weight entries, rebuild structure."""
        with torch.no_grad():
            self.weight.data[~self.sparsity_mask] = 0.0
        self._build_sparse_structure()

    def _set_mask(self, mask: torch.Tensor) -> None:
        """Update sparsity mask and re-zero inactive weight entries."""
        self.sparsity_mask = mask
        self._apply_mask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: y = SpMM(W * mask, x) + bias."""
        return SparsePropLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.sparsity_mask,
            self.w_ptr,
            self.w_col,
            self.w_ptr_csc,
            self.w_row,
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, sparsity: float = 0.75
    ) -> "SparsePropLinear":
        """Convert an existing nn.Linear to SparsePropLinear.

        Copies weight/bias and applies a random sparsity mask. All weight
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
            # Fresh random mask instead of all-zero default
            mask = torch.rand_like(linear.weight) > sparsity
            # Enforce >=1 nnz per row
            for i in range(mask.shape[0]):
                if not mask[i].any():
                    mask[i, int(torch.randint(mask.shape[1], (1,)).item())] = True
            module.sparsity_mask = mask
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
        from nanochat.lcqat.linear import LCQATLinear

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
        self.weight_quantizer = lcqat_linear.weight_quantizer
        self.act_quantizer = lcqat_linear.act_quantizer
        # NOTE: do NOT keep a reference to lcqat_linear as an attribute —
        # nn.Module.__setattr__ would register it as a submodule, re-registering
        # its weight/bias/quantizers under a second path and recreating the
        # orphan + duplicate-parameter bug. The weight/bias/quantizers above
        # are the only references needed; lcqat_linear is left to be GC'd.
        # Apply static sparsity mask (Sparse Transfer) on the shared weight.
        _init_sparsity_mask(self, sparsity)

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
        x_q = self.act_quantizer(x_flat)
        w_q = self.weight_quantizer(self.weight)
        W_eff = w_q.value * self.sparsity_mask.to(w_q.value.dtype)

        # Sparse matmul via autograd Function (weight here is W_eff, which carries
        # the STE graph back to codebook params)
        bias = self.bias if self.bias is not None else None
        out = SparsePropLinearFunction.apply(
            x_q.value,
            W_eff,
            bias,
            self.sparsity_mask,
            self.w_ptr,
            self.w_col,
            self.w_ptr_csc,
            self.w_row,
        )

        if orig_shape[:-1] != out.shape[:-1] or out.shape[-1] != self.out_features:
            out = out.reshape(*orig_shape[:-1], self.out_features)
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
) -> None:
    """Initialize sparsity masks on all SparsePropLinear modules.

    Args:
        model: model to mask
        sparsity: fraction to prune (0.0-1.0)
        target_modules: optional name suffixes to limit scope
    """
    for name, module in model.named_modules():
        if isinstance(module, SparsePropLinear):
            if target_modules is not None:
                if not any(name.endswith(suffix) for suffix in target_modules):
                    continue
            _init_sparsity_mask(module, sparsity)


def _init_sparsity_mask(module: SparsePropLinear, sparsity: float) -> None:
    """Random sparsity mask with >=1 nnz per row."""
    with torch.no_grad():
        mask = torch.rand(module.weight.shape, device=module.weight.device) > sparsity
        row_sums = mask.sum(dim=1)
        for i in range(mask.shape[0]):
            if row_sums[i] == 0:
                j = int(
                    torch.randint(
                        mask.shape[1], (1,), device=module.weight.device
                    ).item()
                )
                mask[i, j] = True
        module.sparsity_mask.copy_(mask)
        module._apply_mask()


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
    from nanochat.lcqat.linear import LCQATLinear

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
    "SparsePropLinear",
    "SparsePropLinearLCQAT",
    "SparsePropLinearFunction",
    "apply_static_sparsity_mask",
    "inject_sparseprop_layers",
    "_init_sparsity_mask",
]
