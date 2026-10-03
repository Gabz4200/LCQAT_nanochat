"""Tiny, fully-materialized models for the paired LC-QAT ablation experiments.

These builders live here, not in `tests/`, because the shipped ablation harness
imports them. A package that imports its own test suite is importable only when
the repository root happens to be on `sys.path`, which is true under pytest
(`pythonpath = ["."]`) and false for an installed wheel.

They deliberately build *active* models rather than plain random ones: nanochat
zero-initializes `attn.c_proj` and `mlp.c_proj`, so an untouched tiny GPT has
attention and MLP outputs of exactly zero and every KV-cache or runtime-parity
assertion over it passes vacuously.
"""

from __future__ import annotations

import torch

from nanochat.models.backbone import GPT, GPTConfig
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)


def build_tiny_gpt() -> GPT:
    """Materialize a small CPU GPT through the standard meta -> to_empty -> init flow."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def build_active_tiny_gpt() -> GPT:
    """Tiny GPT whose zero-initialized projections are randomized.

    Randomized BEFORE any retrofit, so the quantizer initializers see real
    weight ranges rather than a degenerate all-zero span.
    """
    model = build_tiny_gpt()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("c_proj.weight"):
                param.normal_(std=0.02)
    return model


def _fresh_like(template: torch.nn.Module) -> torch.nn.Module:
    """A deep copy of `template` with independently randomized weights.

    Shape is inherited from the template rather than recomputed, which is the
    point: these structures have non-obvious dimensions (the ResFormer value
    embedding is vocab-sized, the ve_gate is vocab-channel-sized).
    """
    import copy

    module = copy.deepcopy(template)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.02)
    return module


def _ve_gate_like(model) -> torch.nn.Module:
    """A `ve_gate` with the right shape, taken from any block that has one."""
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            return block.attn.ve_gate
    raise AssertionError("no block in the model has a ve_gate to copy the shape from")


def _fresh_block_like(model, layer_idx: int) -> torch.nn.Module:
    """Build a fresh Block for `layer_idx`, with its `ve_gate` matching the depth.

    Uses the real `Block` constructor so the value-embedding / gate pairing is
    derived by the same `has_ve(layer_idx, n_layer)` rule the model uses, instead
    of being copied from a block that was built for a different index.
    """
    from nanochat.models.backbone import Block

    block = Block(model.config, layer_idx)
    block.to_empty(device=next(model.transformer.h[0].parameters()).device)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(std=0.02)
    return block


def resize_to(model: torch.nn.Module, n_layer: int) -> None:
    """Grow or shrink `transformer.h` to `n_layer` blocks, fixing the per-layer scalars.

    `GPT.forward` indexes `resid_lambdas` / `x0_lambdas` by layer, so the layer
    count and those two vectors have to stay in step. The lambda shapes are the
    only thing that depends on depth, so this is a one-line fixup rather than a
    model rebuild.
    """
    # Blocks are rebuilt when growing, because a Block's `ve_gate` exists only on
    # layers that carry a value embedding (`has_ve(layer_idx, n_layer)`), and
    # that predicate depends on the total depth. Cloning a block to a new index
    # therefore has to re-derive the gate, not copy it -- otherwise the
    # gate/value-embedding pairing breaks and GPT.forward calls `None`.
    existing = list(model.transformer.h)
    n_old = len(existing)
    if n_layer < n_old:
        model.transformer.h = torch.nn.ModuleList(existing[:n_layer])
    else:
        rebuilt = existing[:n_layer]
        for i in range(n_old, n_layer):
            rebuilt.append(_fresh_block_like(model, i))
        model.transformer.h = torch.nn.ModuleList(rebuilt)
    # `GPT.forward` indexes several per-layer structures by index, so a depth
    # change has to resize all of them, not just the lambdas. These are lists /
    # ModuleDicts built in `__init__`, so they are cheap to extend or truncate.
    for attr, fill in (("resid_lambdas", 1.0), ("x0_lambdas", 0.0)):
        setattr(
            model,
            attr,
            torch.nn.Parameter(torch.full((n_layer,), fill, dtype=torch.float32)),
        )
    window = model.window_sizes[0]
    model.window_sizes = [window for _ in range(n_layer)]
    # Value embeddings are keyed by layer index string. Copy the shape from an
    # existing one rather than recomputing it: the ResFormer value embedding is
    # sized off the *vocab* dimension, so the correct width is not derivable from
    # n_embd / n_head alone, and getting it wrong fails deep inside GPT.forward
    # with a shape error rather than at construction. Which layers get one is
    # re-derived from the same `has_ve` rule via `_fresh_block_like`. Keep one
    # surviving template to copy shapes from *before* pruning, then prune: a
    # depth change can leave zero eligible layers, and taking the template first
    # means the copy source always exists.
    template = next((m for k, m in model.value_embeds.items() if k.isdigit()), None)
    for i in list(model.value_embeds.keys()):
        if not i.isdigit() or int(i) >= n_layer:
            del model.value_embeds[i]
    if template is not None:
        for i in range(n_layer):
            has_ve = i % 2 == (n_layer - 1) % 2
            if has_ve and str(i) not in model.value_embeds:
                model.value_embeds[str(i)] = _fresh_like(template)
            elif not has_ve and str(i) in model.value_embeds:
                del model.value_embeds[str(i)]
    assert model.resid_lambdas.shape[0] == n_layer
    assert len(model.window_sizes) == n_layer
    # Value embeddings and their gates must agree. GPT.forward calls `ve_gate`
    # whenever `ve` is not None, so a layer with a value embedding but no gate
    # fails at forward time with `NoneType is not callable`. Retained blocks keep
    # whatever gate the *original* depth gave them, which is wrong after a depth
    # change (`has_ve` depends on n_layer), so re-derive it for every layer.
    for i, block in enumerate(model.transformer.h):
        has_ve = i % 2 == (n_layer - 1) % 2
        block.attn.ve_gate = _fresh_like(_ve_gate_like(model)) if has_ve else None
    for i, block in enumerate(model.transformer.h):
        has_ve = str(i) in model.value_embeds
        has_gate = block.attn.ve_gate is not None
        assert has_ve == has_gate, f"layer {i}: ve/gate mismatch"
        if has_ve:
            assert model.value_embeds[str(i)].weight.shape[0] == model.config.vocab_size


def make_engine(
    num_blocks: int = 2, n_layer: int | None = None, active: bool = True
) -> DiffusionBlockEngine:
    """Build an engine over a tiny GPT.

    `active=True` randomizes the zero-initialized projections *and* the engine's
    own zero-initialized modules (denoise heads, adapter output layer). All of
    those are zero at init by design, and a zero tensor transmits no gradient,
    so without this every "does the gradient reach X" assertion would pass
    vacuously.
    """
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    if n_layer is not None and n_layer != len(model.transformer.h):
        resize_to(model, n_layer)
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=num_blocks), dtype=torch.float32
    )
    if active:
        with torch.no_grad():
            for head in engine.denoise_heads:
                head.weight.normal_(std=0.05)
            for adapter in engine.adapters:
                adapter.mlp[-1].weight.normal_(std=0.05)
    return engine


__all__ = [
    "build_active_tiny_gpt",
    "build_tiny_gpt",
    "make_engine",
    "resize_to",
]
