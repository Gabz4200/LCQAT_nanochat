"""Block isolation: the invariant the memory argument rests on.

Only the active block's parameters -- its transformer layers, its AdaLN adapter
and its own denoise head -- receive gradients. That is where the B-fold
activation-memory reduction comes from: gradients exist for L/B layers, not L.

`make_engine(active=True)` randomizes the zero-initialized modules (denoise
heads, adapter output layers). A zero tensor transmits no gradient, so without
that, every "does the gradient reach X" assertion would pass vacuously -- which
is exactly what happened before it was split out.
"""

import torch

from nanochat.models.quant.efqat import SelectiveFreezer
from nanochat.training.diffusion_blocks import _layer_groups
from tests.test_dbcpu_engine import make_engine


def _grad_names(engine):
    return {n for n, p in engine.named_parameters() if p.grad is not None}


def test_when_denoise_step_then_only_the_active_block_receives_gradients():
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    loss, _ = engine.denoise_step(idx, block_idx=1)
    loss.backward()
    got = _grad_names(engine)
    assert any(n.startswith("transformer.h.1.") for n in got)
    assert not any(n.startswith("transformer.h.0.") for n in got)
    assert any(n.startswith("db_adapters.1.") for n in got)
    assert not any(n.startswith("db_adapters.0.") for n in got)
    assert any(n.startswith("db_denoise_heads.1.") for n in got)
    assert not any(n.startswith("db_denoise_heads.0.") for n in got)


def test_when_every_block_is_visited_then_all_blocks_receive_gradients():
    """Over B steps, every block's own params must get exercised."""
    engine = make_engine(3, n_layer=6)
    idx = torch.randint(0, 128, (2, 16))
    for b in range(3):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=b)
        loss.backward()
        got = _grad_names(engine)
        assert any(n.startswith(f"db_denoise_heads.{b}.") for n in got)
        assert any(n.startswith(f"db_adapters.{b}.") for n in got)


def test_when_only_the_active_block_runs_then_grad_count_is_flat_in_depth():
    """Activation memory is O(L/B), so the number of *active* layers is L/B."""
    engine = make_engine(4, n_layer=8)
    idx = torch.randint(0, 128, (2, 16))
    for b in range(4):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=b)
        loss.backward()
        active = _grad_names(engine)
        n_layers = len(
            {n.split(".")[2] for n in active if n.startswith("transformer.h.")}
        )
        assert n_layers == 2, f"8 layers / 4 blocks = 2 per block, got {n_layers}"


def test_when_shared_parameters_then_wte_trains_on_every_block():
    """The embedding table is shared across blocks, so it always trains.

    `denoise_step` uses `wte` twice, with deliberately different gradient
    treatment: the noised input keeps the graph (the paper conditions the
    denoiser on clean token embeddings, and those come from `wte`), while the
    regression target is detached. Without the input path the embedding table
    would never receive a gradient, while `generate()` still needs it to build
    the prefix.
    """
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    for b in range(2):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=b)
        loss.backward()
        grad = engine.model.transformer.wte.weight.grad
        # Non-zero, not merely present: this file's whole premise is that a
        # zero tensor transmits no gradient, so `grad is not None` alone is
        # satisfied by the exact vacuity it exists to rule out.
        assert grad is not None
        assert grad.abs().sum() > 0, f"wte received an all-zero gradient on block {b}"


def test_when_edm_objective_then_lm_head_is_not_in_the_graph():
    """Documents a real limitation rather than papering over it.

    The EDM objective trains a denoiser that predicts a clean *embedding*
    (`denoise_head`), never tokens, so `lm_head` is not on the backward path at
    all and receives no gradient under `--db-objective edm`. It is still used at
    sampling time (`DiffusionBlockEngine.generate`), so an EDM-only run leaves it
    at its initialization.

    This is a property of the objective, not a wiring bug: the loss has no
    logits to attach a gradient to. `--db-objective ce` does train `lm_head`, and
    the two objectives must not be mixed within a run. Pinned so that anyone
    changing the EDM objective notices this dependency.
    """
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))
    engine.zero_grad(set_to_none=True)
    loss = engine.train_step(idx, targets, block_idx=0)
    loss.backward()
    assert engine.model.lm_head.weight.grad is not None, (
        "the CE objective does train it"
    )

    engine.zero_grad(set_to_none=True)
    loss, _ = engine.denoise_step(idx, block_idx=0)
    loss.backward()
    assert engine.model.lm_head.weight.grad is None, (
        "if this now passes, the EDM objective gained a token-level loss and "
        "the sampling path needs re-checking"
    )


def test_when_layer_groups_then_they_partition_in_order():
    assert _layer_groups(5, 3) == [[0, 1], [2, 3], [4]]
    assert _layer_groups(6, 3) == [[0, 1], [2, 3], [4, 5]]
    assert _layer_groups(4, 4) == [[0], [1], [2], [3]]
    for groups in (_layer_groups(7, 3), _layer_groups(9, 4)):
        flat = [i for g in groups for i in g]
        assert flat == list(range(len(flat))), "groups must partition in order"
        sizes = [len(g) for g in groups]
        assert max(sizes) - min(sizes) <= 1, "groups must be balanced"


def test_when_freezer_freezes_then_the_veto_survives_later_steps():
    """EfQAT regression: the freeze must not be undone on the next step.

    `_activate_block` used to re-enable every `transformer.h.*` parameter
    unconditionally, so `efqat_freezer.update(step)` at step N was silently
    reverted at step N+1 and EfQAT never took effect at all.
    """
    engine = make_engine(2, n_layer=4)
    freezer = SelectiveFreezer(engine.model, warmup_steps=0, freeze_middle_frac=1.0)
    # 4 layers: the band is clamped to layers 1..2 (never the first or last).
    n_layer, start, end = freezer.layer_bounds()
    # n_layer is pinned as well: `start`/`end` are only bounds if the band
    # was computed against the depth we actually built.
    assert n_layer == 4 and start >= 1 and end <= 3
    assert freezer.freeze() > 0
    engine.set_freezer(freezer)

    frozen = {
        n
        for n, p in engine.model.named_parameters()
        if n.startswith("transformer.h.") and not p.requires_grad
    }
    assert frozen, "the freezer should have frozen some middle layers"

    for _ in range(3):
        engine._activate_block(0)

    still_frozen = {
        n
        for n, p in engine.model.named_parameters()
        if n.startswith("transformer.h.") and not p.requires_grad
    }
    assert frozen <= still_frozen, "EfQAT freeze was undone by block activation"
    # And the active block's own layers stay trainable.
    assert any(p.requires_grad for p in engine.model.transformer.h[0].parameters())


def test_when_no_freezer_then_block_activation_is_unchanged():
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
    assert engine.model.transformer.wte.weight.requires_grad
    assert engine.model.lm_head.weight.requires_grad


def test_when_set_freezer_none_then_the_veto_stops_applying():
    """Detaching the freezer removes the veto from future block activations.

    `SelectiveFreezer.freeze()` also flips `requires_grad` directly, but the
    engine's arbiter is what re-applies it each step, so clearing the reference
    is what lets the active block's own (frozen-band) layers train again. Layers
    outside the active block stay off either way, which is block isolation doing
    its job rather than the freezer.
    """
    engine = make_engine(2, n_layer=4)
    assert engine.block_layers() == [[0, 1], [2, 3]]
    freezer = SelectiveFreezer(engine.model, warmup_steps=0, freeze_middle_frac=1.0)
    freezer.freeze()
    engine.set_freezer(freezer)

    # Block 0 owns layers 0 and 1; layer 1 is in the frozen middle band. Not all
    # of it: attn.c_q / attn.c_k are CRITICAL_PATTERNS and stay trainable by
    # design (EfQAT keeps the outlier layers), so check a non-critical param.
    frozen_name = "transformer.h.1.mlp.c_fc.weight"
    critical_name = "transformer.h.1.attn.c_q.weight"
    engine._activate_block(0)
    params = dict(engine.model.named_parameters())
    assert not params[frozen_name].requires_grad, "the veto should keep c_fc frozen"
    assert params[critical_name].requires_grad, "critical layers stay trainable"

    engine.set_freezer(None)
    engine._activate_block(0)
    params = dict(engine.model.named_parameters())
    assert params[frozen_name].requires_grad, "clearing the freezer lifts the veto"
    # Block isolation still holds for the non-active block.
    assert not any(p.requires_grad for p in engine.model.transformer.h[2].parameters())

    # And the freezer's own reversal is idempotent with the current state.
    freezer.unfreeze()
    assert all(p.requires_grad for p in engine.model.transformer.h[1].parameters())
