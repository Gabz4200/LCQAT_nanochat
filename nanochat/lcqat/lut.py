"""
Fused activation LUT compilation (LC-QAT PRD section 6).

Compiles a continuous activation f(x) between two quantized layers into a
K_in -> K_out index-to-index table, replacing the runtime activation + bucketize
with a zero-FLOP index gather.
"""

from collections.abc import Callable

import torch


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
    midpoints = (output_codebook[:-1] + output_codebook[1:]) * 0.5
    target_dtype = torch.uint8 if output_codebook.numel() <= 255 else torch.int32
    return torch.bucketize(transformed, midpoints).to(target_dtype)
