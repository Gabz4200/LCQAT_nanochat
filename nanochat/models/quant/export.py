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

from nanochat.models.quant.activation import ACT_BODY_PWL, ACT_BODY_SMOOTHPWL, SmoothPWL
from nanochat.models.quant.learnable_lut import (
    RELAXATION_LOGITS,
    LearnableIndexLut,
)
from nanochat.models.quant.linear import LCQATLinear
from nanochat.models.quant.lut import ACTIVATION_LUTS, compile_activation
from nanochat.models.quant.packing import index_dtype_for_k, pack_weight_indices
from nanochat.models.quant.sparse_artifact import pack_sparse_plan, plan_sparse_export


def _activation_pairs(model: nn.Module, table: tuple):
    """Yield `(module, parent_name, substr, act_name, kwargs)` for every wiring site.

    Both the learnable and the frozen attachment walk the model the same way:
    find an out-quantized LCQATLinear whose name ends in a declared activation
    substring, then resolve the quantized sibling that consumes its output.
    Keeping the walk in one generator is what stops the two attachment modes
    from disagreeing about which layers get a table.
    """
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
            yield module, sibling, act_name, kwargs


@torch.no_grad()
def attach_learnable_activation_luts(
    model: nn.Module,
    relaxation: str = RELAXATION_LOGITS,
    act_body: str = ACT_BODY_PWL,
    activations: tuple | None = None,
) -> int:
    """Attach a *trained* activation table to each quantized layer pair (D9).

    `wire_activation_luts` installs a frozen `compile_activation` table, and
    nothing in the training forward ever reads it: `gpt.py`'s MLP does
    `F.relu(x).square()` in float and quantizes on the other side. So the
    activation a model is trained against and the activation a model is
    exported with are two different functions, which shows up only as an
    accuracy gap after a long training run. That is the mismatch this closes.

    Both arguments default to the shipped behaviour, and
    `RELAXATION_LOGITS` with a `SmoothPWL`-fitted body is the same starting
    table as the bake -- `LearnableIndexLut` seeds itself from
    `compile_activation_lut` -- so attaching it with the default flags is a
    no-op on the value, and only makes the table *trainable*.

    `act_body="smoothpwl"` fits `SmoothPWL` to the same function and bakes it
    through the same codebooks, so the artifact is still one integer
    `K_in -> K_out` table. The body is a training-time parameterization; it
    never reaches inference.

    Returns the number of tables attached.
    """
    table = activations if activations is not None else ACTIVATION_LUTS
    count = 0
    for module, sibling, act_name, kwargs in _activation_pairs(model, table):
        input_codebook = module.out_quantizer.get_codebook()
        output_codebook = sibling.act_quantizer.get_codebook()
        body = None
        if act_body == ACT_BODY_SMOOTHPWL:
            if kwargs:
                raise ValueError(
                    f"--lcqat-act-body=smoothpwl cannot fit activation "
                    f"{act_name!r} with kwargs={kwargs!r}: SmoothPWL takes no "
                    f"activation kwargs. Use --lcqat-act-body=pwl for "
                    f"parameterized activations."
                )
            body = SmoothPWL(
                knots=int(input_codebook.numel()),
                zero_pin=True,
                act_name=act_name,
            ).fit_from_callable()
        lut = LearnableIndexLut(
            input_codebook,
            output_codebook,
            act_name=act_name,
            relaxation=relaxation,
            smooth_body=body,
        )
        module.learnable_activation_lut = lut
        # Install the frozen table too, at the same dtype the non-learnable
        # path would have produced. Hardcoding uint8 truncates for K_act >
        # 255, where `compile_activation` yields int32 and the fused chain's
        # `activation_lut.dtype != torch.uint8` guard then rejects a table
        # this function just built. Bound to a distinct name: reusing `table`
        # here rebinds the loop's iteration variable mid-loop.
        resolved = lut.resolved_table()
        _install_lut(module, resolved.to(index_dtype_for_k(lut.k_out)))
        count += 1
    return count


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
    for module, sibling, act_name, kwargs in _activation_pairs(model, table):
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
        if (
            found_self
            and isinstance(child, LCQATLinear)
            and child.act_quantizer is not None
        ):
            return child
    return None


def _install_lut(module: LCQATLinear, table: torch.Tensor) -> None:
    """Register (or overwrite) the `activation_lut` buffer on `module`."""
    if "activation_lut" in module._buffers:
        module.activation_lut.copy_(table)
    else:
        module.register_buffer("activation_lut", table, persistent=True)


@torch.no_grad()
def export_lcqat_checkpoint(
    model: nn.Module, export_path: str, sparse: bool = True
) -> dict:
    """Export a stripped LC-QAT state_dict and save it with torch.save.

    Mutates the model in place (PRD 8): codebook step parameters are deleted
    and weight parameters are replaced by `packed_weight_indices` buffers
    plus a `weight_index_format` tag (FORMAT_* from nanochat.models.quant.packing).
    The result is an artifact for the quantized inference runtime, not a
    resumable training checkpoint - run it on a model you no longer train.

    `sparse=True` additionally exploits the LC-QAT zero anchor. A SparseProp
    layer stores an exact 0.0 at its pruned positions, and 0.0 is codebook index
    `m_neg`, so the sparse pattern is a subset of the index alphabet rather than
    a separate mask. Those layers are written as CSR over the surviving index
    slots (`sparse_keep_indices` + `sparse_row_ptr` + `sparse_col_indices`) with
    the codebook compacted down to the levels actually referenced. Layers with
    no sparsity mask keep the dense path unchanged.

    Returns the saved state_dict for inspection/testing.
    """
    model.eval()

    for _, module in model.named_modules():
        # `SparsePropLinearLCQAT` re-parents an LCQATLinear's quantizers but is
        # NOT an LCQATLinear subclass, so an isinstance check on LCQATLinear
        # alone silently skips every sparse layer -- the artifact would come out
        # with an FP32 shadow weight intact and no index buffers at all. Match
        # on the quantizer the export actually needs, which both classes have.
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if weight_quantizer is None or not hasattr(module, "K_weight"):
            continue
        # A per-channel layer quantizes through `per_channel_weight_quantizer`,
        # a [C, K] table -- but `weight_quantizer` stays present and readable
        # (LCQATLinear keeps it so K_weight / the optimizer partition still
        # work), so the line below would pack indices from the *shared* table
        # and silently emit an artifact whose indices mean something different
        # from the weights the model was trained with. `LCQATLinear.forward`
        # raises on this same combination at *inference* time; raising here too
        # means the failure lands at export, where the mistake is made, instead
        # of after the shadow weights have already been deleted.
        if getattr(module, "per_channel_weight_quantizer", None) is not None:
            raise ValueError(
                "cannot export a layer using per-channel weight quantization: "
                "the packed-index and fused-LUT inference paths read a single "
                "shared [K] table, which a [C, K] per-channel table cannot "
                "express. Re-export from a model trained without "
                "--lcqat-channel-center, or strip the per-channel table first."
            )
        module.weight_quantizer.compile_for_inference()
        act_quantizer = getattr(module, "act_quantizer", None)
        if act_quantizer is not None:
            act_quantizer.compile_for_inference()
        out_quantizer = getattr(module, "out_quantizer", None)
        if out_quantizer is not None:
            out_quantizer.compile_for_inference()

        indices = module.weight_quantizer(module.weight).indices
        # The dense buffer is always written, and is always packed from the full
        # [m, n] index matrix at the full alphabet. The sparse branch adds the
        # CSR listing beside it; it must not overwrite `packed`/`fmt`, or the
        # dense buffer ends up holding the flat `nnz` sparse values and every
        # dense consumer silently reads a wrong-shaped tensor.
        packed, fmt = pack_weight_indices(indices, module.K_weight)
        module.register_buffer("packed_weight_indices", packed, persistent=True)
        module.register_buffer(
            "weight_index_format", torch.tensor(fmt, dtype=torch.int64), persistent=True
        )

        mask = getattr(module, "sparsity_mask", None)
        if sparse and isinstance(mask, torch.Tensor):
            plan = plan_sparse_export(indices, mask, k=module.K_weight)
            sparse_packed, sparse_fmt = pack_sparse_plan(plan)
            module.register_buffer(
                "sparse_keep_indices", sparse_packed, persistent=True
            )
            module.register_buffer("sparse_row_ptr", plan.row_ptr, persistent=True)
            module.register_buffer(
                "sparse_col_indices", plan.col_indices, persistent=True
            )
            module.register_buffer(
                "sparse_index_format",
                torch.tensor(sparse_fmt, dtype=torch.int64),
                persistent=True,
            )
            module.register_buffer(
                "sparse_alphabet",
                module.weight_quantizer.get_codebook()[plan.used].contiguous(),
                persistent=True,
            )
            module.register_buffer(
                "sparse_k_used",
                torch.tensor(plan.k_used, dtype=torch.int64),
                persistent=True,
            )
        del module.weight

    wire_activation_luts(model)
    state_dict = model.state_dict()
    torch.save(state_dict, export_path)
    return state_dict
