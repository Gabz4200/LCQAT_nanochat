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
from nanochat.modules.experiments.tiny_models import (  # noqa: F401  (re-export)
    build_active_tiny_gpt,
    make_engine,
    resize_to,
)
from nanochat.training.diffusion_blocks import (
    _layer_groups,
)


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
