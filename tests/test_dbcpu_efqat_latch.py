"""EfQAT per-block *permanent* freezing (PRD 3.2).

The middle-band `SelectiveFreezer` is global and one-shot: it freezes a
contiguous band of transformer layers once and only when the global step passes
a warmup. Under DiffusionBlocks the interesting question is per block -- a block
trains on one disjoint equi-probability sigma range, so it converges early and
should then be *retired*. "Retired" has to mean permanent: a later optimizer
step that samples that block again must not move its quantization parameters.

The mechanism that makes it permanent is ordering, not bookkeeping.
`DiffusionBlockEngine._requires_grad_for` consults the freezer *before*
`_activate_block` writes `requires_grad`, so the veto wins every time regardless
of which block is sampled. Each test below therefore has to hold after many
activations *and* across every block index, and each fails if the latch is
replaced by a no-op.
"""

import json

import pytest
import torch

from nanochat.lcqat.efqat import BlockLatchFreezer
from tests.test_dbcpu_engine import make_engine


def _latched_params(engine, freezer, block_idx):
    """Names of `block_idx`'s params that currently refuse gradients."""
    return {
        name
        for name, p in engine.named_parameters()
        if freezer.block_of(name) == block_idx and not p.requires_grad
    }


def _block_params(engine, freezer, block_idx):
    return {
        name
        for name, _ in engine.named_parameters()
        if freezer.block_of(name) == block_idx
    }


def test_when_block_latched_then_stays_frozen_across_later_steps_and_blocks() -> None:
    """The core claim: a latch is one-way, even when its block is resampled.

    Every other block is visited repeatedly on purpose. Block isolation alone
    would make block 0 look frozen whenever block 1 is active, which is how a
    non-permanent freeze passes a naive test: sample the latched block, and the
    parameters must still be frozen.
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)

    assert freezer.latch_block(1) > 0
    expected = _block_params(engine, freezer, 1)
    assert expected, "block 1 should own parameters (layers, adapter, head)"

    idx = torch.randint(0, 128, (2, 16))
    # 30 steps over all three blocks, block 1 included -- the whole point is
    # that sampling block 1 does not revive it.
    for step in range(30):
        block = step % 3
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=block)
        loss.backward()
        assert not _latched_params(engine, freezer, 1) ^ expected, (
            f"step {step} (block {block}) changed block 1's frozen set"
        )
        assert all(
            not p.requires_grad
            for name, p in engine.named_parameters()
            if name in expected
        ), f"step {step} (block {block}) revived a latched parameter"

    assert freezer.is_latched(1)


def test_when_latched_block_is_the_sampled_block_then_no_grad_reaches_it() -> None:
    """A latched block's parameters must receive no gradient, not just be flagged.

    `requires_grad=False` alone would be enough for AdamW to skip them, so this
    test also pins the observable consequence: the latched block's own params
    produce no `.grad` even when it is the active block. Uses a real backward,
    so it fails if the latch is dropped even by a path that leaves the flag set.
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    freezer.latch_block(2)

    idx = torch.randint(0, 128, (2, 16))
    engine.zero_grad(set_to_none=True)
    loss, _ = engine.denoise_step(idx, block_idx=2)
    loss.backward()

    latched = _block_params(engine, freezer, 2)
    assert latched
    got = {n for n, p in engine.named_parameters() if p.grad is not None}
    assert not (latched & got), f"gradients leaked into latched block: {latched & got}"


def test_when_never_latched_block_sampled_then_it_still_receives_gradients() -> None:
    """Latching one block must not leak into its neighbours or shared params."""
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    freezer.latch_block(0)

    idx = torch.randint(0, 128, (2, 16))
    for block in (1, 2):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=block)
        loss.backward()
        got = {n for n, p in engine.named_parameters() if p.grad is not None}
        own = _block_params(engine, freezer, block)
        assert own & got, f"block {block} lost its gradients"
        assert not any(n.startswith("db_denoise_heads.0.") for n in got)
    # Embeddings/lm_head are shared by every block, so no single latch can claim
    # them; they must stay trainable or the model stops learning entirely.
    assert engine.model.transformer.wte.weight.requires_grad
    assert engine.model.lm_head.weight.requires_grad


def test_when_latch_state_then_readable_and_round_trips_through_metadata() -> None:
    """Latch state must be readable and survive into checkpoint metadata.

    Metadata only -- tensors are already in the state_dict -- but the freeze
    decision itself is not recoverable from tensor values: a frozen parameter is
    indistinguishable from a converged one. Without this, a resumed run would
    silently start training a block the previous run had retired.
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    assert freezer.latched_blocks() == []
    assert not freezer.is_latched(0)

    freezer.latch_blocks([0, 2])
    meta = engine.freezer_metadata()
    assert meta["latched_blocks"] == [0, 2]
    assert meta["n_blocks"] == 3
    assert freezer.is_latched(2) and not freezer.is_latched(1)
    # Must be JSON-serializable: save_checkpoint json.dumps the metadata.
    assert json.loads(json.dumps(meta))["latched_blocks"] == [0, 2]

    # A fresh run resuming from that checkpoint inherits the latch.
    resumed = BlockLatchFreezer(engine, engine.block_layers())
    assert resumed.latched_blocks() == []
    resumed.load_metadata(json.loads(json.dumps(meta)))
    assert resumed.latched_blocks() == [0, 2]
    assert resumed.is_latched(0)
    # And the resumed latch actually gates: block 2 is frozen, block 1 is not.
    resumed_engine = make_engine(3, n_layer=6)
    resumed = BlockLatchFreezer(resumed_engine, resumed_engine.block_layers())
    resumed.load_metadata(meta)
    resumed_engine.set_freezer(resumed)
    resumed_engine._activate_block(2)
    # Loading the metadata must actually freeze the tensors, not just record
    # the index -- otherwise a resumed run reports "latched" while still training.
    assert _latched_params(resumed_engine, resumed, 2) == _block_params(
        resumed_engine, resumed, 2
    )
    resumed_engine._activate_block(1)
    trainable = {nm for nm, p in resumed_engine.named_parameters() if p.requires_grad}
    assert not any(n.startswith("db_denoise_heads.2.") for n in trainable)
    assert any(n.startswith("db_denoise_heads.1.") for n in trainable)


def test_when_latch_block_called_twice_then_idempotent() -> None:
    """Re-latching (every step after the threshold) must not double-count."""
    engine = make_engine(2, n_layer=4)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    first = freezer.latch_block(0)
    assert first > 0
    assert freezer.latch_block(0) == 0
    assert freezer.metadata()["latched_blocks"] == [0]


def test_when_all_blocks_latched_then_sampling_raises_rather_than_returning_one() -> (
    None
):
    """Training over must be loud, not a silent no-op step.

    Sampling a retired block yields a loss with no `grad_fn` (the EDM objective
    detaches the clean-embedding input), so the backward crashes. Returning an
    index anyway would turn a clear "stop" into an obscure `RuntimeError` deep
    in autograd.
    """
    engine = make_engine(2, n_layer=4)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    freezer.latch_blocks([0, 1])
    assert engine.live_blocks() == []
    with pytest.raises(RuntimeError, match="nothing left to train"):
        engine.sample_block()


def test_when_sampling_then_a_latched_block_is_never_drawn() -> None:
    """The sampler is the only defence against the no-`grad_fn` backward.

    Latching alone is not enough: a latched block stays in the partitioner's
    index range, so without filtering the sampler would keep drawing it and the
    run would crash. Drawn many times, because a single draw proves nothing about
    a filter that is off by one.
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    freezer.latch_blocks([1])
    assert engine.live_blocks() == [0, 2]

    drawn = {engine.sample_block() for _ in range(200)}
    assert drawn == {0, 2}
    assert 1 not in drawn


def test_when_sampling_a_live_block_then_the_step_has_a_grad_fn() -> None:
    """End-to-end proof of the fix: the sampled step is actually differentiable.

    This is the failure the smoke run surfaced -- a latched block's `denoise_step`
    returns a tensor with `grad_fn=None` and `backward()` then raises. Asserting
    on `live_blocks()` alone would pass even if `_activate_block` reintroduced
    the block, so the real backward is what is checked.
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    freezer.latch_blocks([1])
    idx = torch.randint(0, 128, (2, 16))

    for _ in range(10):
        block_idx = engine.sample_block()
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=block_idx)
        assert loss.grad_fn is not None, f"block {block_idx} produced no grad_fn"
        loss.backward()
        got = {n for n, p in engine.named_parameters() if p.grad is not None}
        assert _block_params(engine, freezer, block_idx) & got


def test_when_no_latch_then_engine_train_behaviour_is_unchanged() -> None:
    """With the freezer attached but nothing latched, gating must be inert.

    Guards against a freezer that vetoes by accident (e.g. a name->block
    resolution that maps shared params onto block 0).
    """
    engine = make_engine(3, n_layer=6)
    freezer = BlockLatchFreezer(engine, engine.block_layers())
    engine.set_freezer(freezer)
    idx = torch.randint(0, 128, (2, 16))
    engine.zero_grad(set_to_none=True)
    loss, _ = engine.denoise_step(idx, block_idx=1)
    loss.backward()
    got = {n for n, p in engine.named_parameters() if p.grad is not None}
    assert _block_params(engine, freezer, 1) & got
    # Shared params are attributed to no block and must survive untouched.
    assert engine.model.transformer.wte.weight.requires_grad
    assert engine.model.lm_head.weight.requires_grad
    assert freezer.block_of("transformer.wte.weight") is None
    assert freezer.block_of("lm_head.weight") is None
