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

from nanochat.lcqat.linear import LCQATLinear
from nanochat.lcqat.lut import ACTIVATION_LUTS, compile_activation
from nanochat.lcqat.packing import pack_weight_indices


@torch.no_grad()
def wire_activation_luts(model: nn.Module, activations: tuple | None = None) -> int:
    """Compile elementwise activation tables for quantized-inference layer pairs.

    "Activation Functions become LUTs too" (PRD section 6): every activation
    sitting between two quantized LCQATLinear layers is compiled into an
    index->index LUT stored on the first layer (`activation_lut`), so the
    runtime replaces float math with a zero-FLOP gather.

    The default `ACTIVATION_LUTS` table wires `relu^2` between every MLP
    `c_fc` (out-quantized) and its `c_proj` (act-quantized). Extra activations
    (SiLU, GELU, tanh, ...) are picked up by adding entries of the form
    `(module_name_substring, activation_name, kwargs)`.

    Returns the number of tables wired.
    """
    table = activations if activations is not None else ACTIVATION_LUTS
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LCQATLinear) or module.out_quantizer is None:
            continue
        for substr, act_name, kwargs in table:
            if not name.endswith(substr):
                continue
            # Parent is the dotted path up to (but excluding) the leaf suffix.
            parent_name = name[: -len(substr)]
            parent_name = parent_name[:-1] if parent_name.endswith(".") else parent_name
            sibling = _next_quantized_sibling(model, parent_name, substr.split(".")[-1])
            if sibling is None or sibling.act_quantizer is None:
                continue
            lut = compile_activation(
                act_name,
                module.out_quantizer.get_codebook(),
                sibling.act_quantizer.get_codebook(),
                kwargs,
            )
            _install_lut(module, lut)
            count += 1
    return count


def _next_quantized_sibling(
    model: nn.Module, parent_name: str, leaf: str
) -> LCQATLinear | None:
    """Find the quantized Linear sibling that follows `leaf` under `parent_name`.

    nanochat's MLP is `c_fc -> relu^2 -> c_proj` (leaf names `c_fc` and `c_proj`
    under the same parent). Walks the parent's registered child modules in
    declaration order and returns the first LCQATLinear with an act_quantizer
    whose key comes after `leaf`.
    """
    parent = model.get_submodule(parent_name) if parent_name else model
    found_self = False
    for attr, child in parent.named_children():
        if attr == leaf:
            found_self = True
            continue
        if found_self and isinstance(child, LCQATLinear) and child.act_quantizer is not None:
            return child
    return None


def _install_lut(module: LCQATLinear, table: torch.Tensor) -> None:
    """Register (or overwrite) the `activation_lut` buffer on `module`."""
    if "activation_lut" in module._buffers:
        module.activation_lut.copy_(table)
    else:
        module.register_buffer("activation_lut", table, persistent=True)


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
