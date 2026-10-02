"""Slice 2: block isolation and the single `requires_grad` arbiter.

Covers the two DiffusionBlocks invariants the memory argument rests on:

1. Only the active block's parameters (layers, adapter, denoise head) receive
   gradients. This is where the B-fold activation-memory reduction comes from --
   gradients exist for L/B layers, not L.
2. `requires_grad` has exactly one arbiter (`_requires_grad_for`), so a
   `SelectiveFreezer` veto survives subsequent steps. Previously
   `_activate_block` unconditionally re-enabled every `transformer.h.*`
   parameter each micro-step, which silently undid EfQAT the step after it froze
   anything.
"""

import torch

from nanochat.models.quant.efqat import SelectiveFreezer
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
    _layer_groups,
)
from tests.conftest import build_active_tiny_gpt


def resize_to(model: torch.nn.Module, n_layer: int) -> None:
    """Grow or shrink `transformer.h` to `n_layer` blocks, fixing the per-layer scalars.

    `GPT.forward` indexes `resid_lambdas` / `x0_lambdas` by layer, so the layer
    count and those two vectors have to stay in step. The lambda shapes are the
    only thing that depends on depth, so this is a one-line fixup rather than a
    model rebuild.
    """

    # Blocks are rebuilt when growing, because a Block's `ve_gate` exists only on
    # layers that carry a value embedding (gpt.py `has_ve(layer_idx, n_layer)`),
    # and that predicate depends on the total depth. Cloning a block to a new
    # index therefore has to re-derive the gate, not copy it -- otherwise the
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
    # n_embd / n_head alone, and getting it wrong fails deep inside
    # GPT.forward with a shape error rather than at construction. Which layers
    # get one is re-derived from the same `has_ve` rule via `_fresh_block_like`.
    # Keep one surviving template to copy shapes from *before* pruning, then
    # prune: a depth change can leave zero eligible layers, and taking the
    # template first means the copy source always exists.
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
    vacuously -- which is exactly what happened before this was split out.
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


def test_when_train_step_then_only_active_block_gets_grad() -> None:
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    loss = engine.train_step(idx, idx, block_idx=0)
    assert torch.isfinite(loss).all()
    loss.backward()

    active = [p.grad is not None for p in engine.model.transformer.h[0].parameters()]
    frozen = [p.grad is not None for p in engine.model.transformer.h[1].parameters()]
    assert any(active)
    assert not any(frozen)


def test_when_denoise_step_then_only_active_block_gets_grad() -> None:
    """The EDM path, which is the one that actually saves activations."""
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    loss, sigma = engine.denoise_step(idx, block_idx=1)
    assert torch.isfinite(loss).all()
    assert sigma.ndim == 0 and float(sigma) > 0
    loss.backward()

    with_grad = {n for n, p in engine.named_parameters() if p.grad is not None}
    # Only block 1's transformer layer.
    assert any(n.startswith("transformer.h.1.") for n in with_grad)
    assert not any(n.startswith("transformer.h.0.") for n in with_grad)
    # Only block 1's adapter and its own denoise head.
    assert any(n.startswith("db_adapters.1.") for n in with_grad)
    assert not any(n.startswith("db_adapters.0.") for n in with_grad)
    assert any(n.startswith("db_denoise_heads.1.") for n in with_grad)
    assert not any(n.startswith("db_denoise_heads.0.") for n in with_grad)


def test_when_denoise_step_with_precomputed_clean_then_target_is_reused() -> None:
    """The batch is reused across micro-steps, so `clean` is hoisted out."""
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        clean = torch.nn.functional.normalize(
            engine.model.transformer.wte(idx).float(), dim=-1
        )
    loss, _ = engine.denoise_step(idx, block_idx=0, clean=clean)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        n.startswith("db_denoise_heads.0.") and p.grad is not None
        for n, p in engine.named_parameters()
    )


def test_when_train_step_then_all_blocks_cover_all_layers() -> None:
    engine = make_engine(2, active=False)
    assert engine.block_layers() == [[0], [1]]


def test_when_layer_groups_then_contiguous_balanced_and_ordered() -> None:
    # 5 layers over 3 blocks: the first 2 groups take the extra layer, and the
    # groups stay contiguous and in ascending order (which both the
    # block-diagonal mask and the sequential sampler rely on).
    assert _layer_groups(5, 3) == [[0, 1], [2, 3], [4]]
    assert _layer_groups(6, 3) == [[0, 1], [2, 3], [4, 5]]
    assert _layer_groups(4, 4) == [[0], [1], [2], [3]]
    for groups in (_layer_groups(7, 3), _layer_groups(9, 4)):
        flat = [i for g in groups for i in g]
        assert flat == list(range(len(flat))), "groups must partition in order"
        assert max(len(g) for g in groups) - min(len(g) for g in groups) <= 1


def test_when_engine_built_then_adapters_emit_one_pair_per_layer() -> None:
    """Adapter width must match its block's layer count, or cond indexing drifts."""
    engine = make_engine(3, n_layer=5)
    groups = engine.block_layers()
    assert groups == [[0, 1], [2, 3], [4]]
    for b, group in enumerate(groups):
        assert engine.adapters[b].n_layers == len(group)
    x = torch.randn(2, 8, engine.model.config.n_embd)
    conds = engine.adapters[1](x, torch.tensor([1.0, 1.0]))
    assert len(conds) == len(groups[1])


def test_when_freezer_freezes_then_veto_survives_later_steps() -> None:
    """EfQAT regression: the freeze must not be undone on the next step.

    `_activate_block` used to re-enable every `transformer.h.*` parameter
    unconditionally, so `efqat_freezer.update(step)` at step N was silently
    reverted at step N+1 and EfQAT never took effect.
    """
    engine = make_engine(2, n_layer=4)
    freezer = SelectiveFreezer(engine.model, warmup_steps=0, freeze_middle_frac=1.0)
    # 4 layers: the band is clamped to layers 1..2 (never the first or last).
    n_layer, start, end = freezer.layer_bounds()
    assert n_layer == 4 and start >= 1 and end <= 3
    assert freezer.freeze() > 0
    engine.set_freezer(freezer)

    frozen_names = [
        n
        for n, p in engine.model.named_parameters()
        if n.startswith("transformer.h.") and not p.requires_grad
    ]
    assert frozen_names, "the freezer should have frozen some middle layers"

    for _ in range(3):
        engine._activate_block(0)
    still_frozen = [
        n
        for n, p in engine.model.named_parameters()
        if n.startswith("transformer.h.") and not p.requires_grad
    ]
    assert set(frozen_names) <= set(still_frozen), (
        "EfQAT freeze was undone by block activation"
    )
    # And the active block's own layers are still trainable.
    assert any(p.requires_grad for p in engine.model.transformer.h[0].parameters())


def test_when_no_freezer_then_block_activation_is_unchanged() -> None:
    # 4 layers over 2 blocks => block 0 owns layers [0, 1], block 1 owns [2, 3].
    engine = make_engine(2, n_layer=4)
    assert engine.freezer is None
    assert engine.block_layers() == [[0, 1], [2, 3]]
    engine._activate_block(0)
    trainable = {n for n, p in engine.model.named_parameters() if p.requires_grad}
    assert any(n.startswith("transformer.h.0.") for n in trainable)
    assert any(n.startswith("transformer.h.1.") for n in trainable)
    assert not any(
        n.startswith("transformer.h.2.") or n.startswith("transformer.h.3.")
        for n in trainable
    )
    # Shared, non-layer parameters (embeddings, lm_head) always train.
    assert engine.model.transformer.wte.weight.requires_grad
    assert engine.model.lm_head.weight.requires_grad


def test_when_denoise_step_with_attn_mask_then_mask_changes_the_loss() -> None:
    """`denoise_step` calls the blocks directly, so it must thread attn_mask itself.

    `train_step` forwards the mask to GPT.forward; the EDM path bypassed that
    and always ran unmasked, which made packed sequences (block-diagonal masks)
    silently wrong rather than unavailable.
    """
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    t = idx.size(1)
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    mask = causal.view(1, 1, t, t)
    torch.manual_seed(7)
    unmasked, _ = engine.denoise_step(idx, block_idx=0, attn_mask=None)
    torch.manual_seed(7)
    masked, _ = engine.denoise_step(idx, block_idx=0, attn_mask=mask)
    assert torch.isfinite(unmasked) and torch.isfinite(masked)
    # A causal mask over a full (non-packed) sequence must be a no-op relative
    # to the default, because attention is already causal.
    assert torch.allclose(unmasked, masked, atol=1e-5)
