"""
LC-QAT checkpoint export (LC-QAT PRD sections 8 and 6).

Strips FP32 shadow weights, freezes codebooks into static FP32 LUTs, and
saves a minimal state_dict of uint8 weight indices + codebook buffers.
"""

import torch
import torch.nn as nn

from nanochat.lcqat.linear import LCQATLinear


@torch.no_grad()
def export_lcqat_checkpoint(model: nn.Module, export_path: str) -> dict:
    """Export a stripped LC-QAT state_dict and save it with torch.save.

    Mutates the model in place (PRD 8): codebook step parameters are deleted
    and weight parameters are replaced by `packed_weight_indices` buffers.
    The result is an artifact for the quantized inference runtime, not a
    resumable training checkpoint - run it on a model you no longer train.

    Returns the saved state_dict for inspection/testing.
    """
    model.eval()

    for _, module in model.named_modules():
        if not isinstance(module, LCQATLinear):
            continue
        module.weight_quantizer.compile_for_inference()
        module.act_quantizer.compile_for_inference()
        if module.out_quantizer is not None:
            module.out_quantizer.compile_for_inference()

        indices = module.weight_quantizer(module.weight).indices
        if indices.numel() and int(indices.max()) > 255:
            raise ValueError(
                f"K_weight={module.K_weight} exceeds uint8 export format (K <= 255)"
            )
        module.register_buffer(
            "packed_weight_indices", indices.to(torch.uint8), persistent=True
        )
        del module.weight

    state_dict = model.state_dict()
    torch.save(state_dict, export_path)
    return state_dict
