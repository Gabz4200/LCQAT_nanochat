"""
LC-QAT checkpoint export (LC-QAT PRD sections 8 and 6).

Strips FP32 shadow weights, freezes codebooks into static FP32 LUTs, and
saves a minimal state_dict of K-selected packed weight indices + codebook
buffers (trits for K=3, nibbles for K<=15, uint8 for K<=255, int32 above),
plus fused activation-LUT tables for the elementwise ops between quantized
layers (PRD section 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.lcqat.linear import LCQATLinear
from nanochat.lcqat.lut import compile_activation_lut
from nanochat.lcqat.packing import pack_weight_indices


@torch.no_grad()
def wire_activation_luts(model: nn.Module) -> int:
    """Compile elementwise activation tables for quantized-inference MLP pairs.

    nanochat's MLP is c_fc (out-quantized) -> relu^2 -> c_proj
    (act-quantized); the composition compiles to an index->index table on
    c_fc (`activation_lut`), which the fused MLP forward consumes instead
    of float math. Returns the number of tables wired. Call before
    load_state_dict on a fresh model so exported state keys match.
    """
    count = 0
    for name, module in model.named_modules():
        if not name.endswith("mlp.c_fc") or not isinstance(module, LCQATLinear):
            continue
        parent = model.get_submodule(name[: -len(".c_fc")])
        next_linear = getattr(parent, "c_proj", None)
        if not isinstance(next_linear, LCQATLinear) or module.out_quantizer is None:
            continue
        table = compile_activation_lut(
            lambda t: F.relu(t).square(),
            module.out_quantizer.get_codebook(),
            next_linear.act_quantizer.get_codebook(),
        )
        if "activation_lut" in module._buffers:
            module.activation_lut.copy_(table)
        else:
            module.register_buffer("activation_lut", table, persistent=True)
        count += 1
    return count


@torch.no_grad()
def export_lcqat_checkpoint(model: nn.Module, export_path: str) -> dict:
    """Export a stripped LC-QAT state_dict and save it with torch.save.

    Mutates the model in place (PRD 8): codebook step parameters are deleted
    and weight parameters are replaced by `packed_weight_indices` buffers
    plus a `weight_index_format` tag (FORMAT_* from nanochat.lcqat.packing).
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
        packed, fmt = pack_weight_indices(indices, module.K_weight)
        module.register_buffer("packed_weight_indices", packed, persistent=True)
        module.register_buffer(
            "weight_index_format", torch.tensor(fmt, dtype=torch.int64), persistent=True
        )
        del module.weight

    wire_activation_luts(model)
    state_dict = model.state_dict()
    torch.save(state_dict, export_path)
    return state_dict
