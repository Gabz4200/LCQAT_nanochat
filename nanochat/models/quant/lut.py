"""
Fused activation LUT compilation (LC-QAT PRD section 6).

Compiles a continuous activation f(x) between two quantized layers into a
K_in -> K_out index-to-index table, replacing the runtime activation + bucketize
with a zero-FLOP index gather.

"Activation Functions become LUTs too" (PRD: every elementwise op sitting
between two quantized layers — relu^2, SiLU, GELU, tanh, sigmoid, softcap-tanh —
is compiled into an index->index table at export time, so the runtime is purely
table fetches + FMA).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from nanochat.models.quant.codebook import codebook_midpoints
from nanochat.models.quant.packing import index_dtype_for_k


@torch.no_grad()
def compile_activation_lut(
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    input_codebook: torch.Tensor,
    output_codebook: torch.Tensor,
) -> torch.Tensor:
    """Compile `act_fn` over `input_codebook` into indices of `output_codebook`.

    Args:
        act_fn: elementwise activation (e.g. relu^2, GELU, SiLU, tanh).
        input_codebook: FP32 codebook of the pre-activation tensor, shape (K_in,).
        output_codebook: FP32 codebook of the post-activation quantizer, shape (K_out,).

    Returns:
        uint8 (K_out <= 255) or int32 tensor of shape (K_in,) holding target
        indices in [0, K_out - 1].
    """
    if input_codebook.ndim != 1 or output_codebook.ndim != 1:
        raise ValueError(
            f"codebooks must be 1-D, got input {tuple(input_codebook.shape)}, "
            f"output {tuple(output_codebook.shape)}"
        )
    if input_codebook.numel() < 3 or output_codebook.numel() < 3:
        raise ValueError("codebooks must have at least 3 entries (odd K >= 3)")

    transformed = act_fn(input_codebook.to(torch.float32))
    midpoints = codebook_midpoints(output_codebook.to(torch.float32))
    target_dtype = index_dtype_for_k(output_codebook.numel())
    return torch.bucketize(transformed, midpoints).to(target_dtype)


# Activation registry: name -> elementwise callable on a 1-D FP32 codebook.
_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu2": lambda x: F.relu(x).square(),
    "silu": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a registered activation name to its callable."""
    if name not in _ACTIVATIONS:
        raise KeyError(
            f"unknown activation {name!r}; registered: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]


def compile_activation(
    name: str,
    input_codebook: torch.Tensor,
    output_codebook: torch.Tensor,
    kwargs: dict | None = None,
) -> torch.Tensor:
    """Compile a named activation into a LUT between two codebooks."""
    fn = get_activation(name)
    if kwargs:
        fn = lambda t, _fn=fn, _kw=kwargs: _fn(t, **_kw)
    return compile_activation_lut(fn, input_codebook, output_codebook)


ACTIVATION_LUTS: tuple[tuple[str, str, dict | None], ...] = (
    # leaf name suffix -> activation name -> kwargs
    ("c_fc", "relu2", None),
)
