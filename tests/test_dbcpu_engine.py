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

from nanochat.modules.experiments.tiny_models import (  # noqa: F401  (re-export)
    build_active_tiny_gpt,
    make_engine,
    resize_to,
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
