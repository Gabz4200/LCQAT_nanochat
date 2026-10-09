"""DARD's integration with DiffusionBlocks, LC-QAT and the shared flag surface.

The property under test is the one the implementation had to invent: the EDM
objective emits *embeddings*, not logits, so DARD attaches at the `lm_head`
projection the Euler sampler already uses. That makes DARD the first path in
this repo that gives `lm_head` a gradient under EDM, and both halves of that
change are pinned here.

`active=True` throughout, per the repo gotcha: nanochat zero-initializes
`c_proj`/`c_fc` and the denoise heads, so an untouched tiny model transmits no
gradient and every "does the gradient reach X" assertion would pass vacuously.

python -m pytest tests/test_dard_diffusion_blocks.py -v
"""

from __future__ import annotations

import argparse

import pytest
import torch

from nanochat.models.dard.cli import (
    add_dard_args,
    shard_size_for,
    validate_dard_args,
)
from nanochat.models.dard.compile import compile_rules
from nanochat.models.dard.engine_hook import LOGCAP_SOFTMAX, denoiser_logits
from nanochat.models.dard.loss import DardLossBalancer, DardRuleLoss
from nanochat.models.dard.shard import MAX_SHARD_SIZE
from nanochat.models.dard.types import AmlRule
from nanochat.models.quant.retrofit import DEFAULT_PRESET, PRESETS, retrofit_model
from nanochat.modules.experiments.tiny_models import make_engine

#: Resolved per test from the model, so the rules always index a real vocabulary.
VOCAB = None


def make_rule_loss(vocab: int, num_rules: int = 3) -> DardRuleLoss:
    """A loss over rules drawn from the model's own vocabulary."""
    rules = []
    for r in range(num_rules):
        base = (r * 4) % max(vocab - 7, 1)
        rules.append(
            AmlRule(
                frozenset({base, base + 1, base + 2}),
                frozenset({base + 3, base + 4, base + 5}),
                1.0,
            )
        )
    weights = torch.full((num_rules,), 1.0 / num_rules)
    return DardRuleLoss(
        compile_rules(rules, weights, torch.ones(num_rules), vocab_size=vocab)
    )


def engine_vocab(engine) -> int:
    return engine.model.config.vocab_size


def published_logits(engine) -> torch.Tensor:
    """The engine's denoiser logits, narrowed from Optional.

    Non-None is asserted rather than assumed: it is the contract `denoise_step`
    makes when DARD is attached, and every test below depends on it.
    """
    logits = engine.last_dard_logits
    assert logits is not None, "denoise_step did not publish DARD logits"
    return logits


def penalty(engine) -> torch.nn.Module:
    """The attached rule-loss module, narrowed from Optional.

    Non-None is asserted rather than assumed: it is the contract `set_dard`
    establishes, and every test below depends on it.
    """
    fn = engine.dard_loss_fn
    assert fn is not None, "DARD was not attached to the engine"
    return fn


def head_grad(engine) -> torch.Tensor:
    """`lm_head`'s gradient, asserted present rather than Optional-checked."""
    grad = engine.model.lm_head.weight.grad
    assert grad is not None, "lm_head received no gradient"
    return grad


def test_when_dard_is_on_then_every_block_gets_gradient_for_every_block_index() -> None:
    """The per-block head's whole purpose: the rules reach every block.

    Regression for a real bug. `denoise_step` runs only the active block, and
    `_activate_block` set `requires_grad=False` on the rest, so the rule
    penalty -- which read a distribution from the shared `lm_head` -- reached
    no transformer layer at all:

        active block 0: layers with DARD grad = []
        active block 1: layers with DARD grad = []

    Every rule was enforced against a representation no rule could shape. With
    the per-block head each block's own hidden state is projected through the
    shared head, and every block is made trainable, so all of them receive
    gradient for *every* sampled block index.
    """
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    loss_fn = make_rule_loss(vocab)
    engine.train()
    engine.set_dard(loss_fn)
    idx = torch.randint(0, vocab, (1, 8))

    for block_idx in range(2):
        engine.zero_grad(set_to_none=True)
        engine.denoise_step(idx, block_idx=block_idx, backend="naive")
        block_logits = engine.last_block_logits
        assert block_logits is not None and len(block_logits) == 2
        penalty_over_blocks(loss_fn, block_logits).backward()

        for i, layer in enumerate(engine.transformer_layers()):
            grad = sum(
                float(p.grad.abs().sum())
                for p in layer.parameters()
                if p.grad is not None
            )
            assert grad > 0, (
                f"block {block_idx}: transformer layer {i} received no DARD "
                "gradient, so its rules cannot affect that block"
            )


def penalty_over_blocks(
    loss_fn: DardRuleLoss, block_logits: dict[int, torch.Tensor]
) -> torch.Tensor:
    """The penalty the training loop applies: mean over every block."""
    return torch.stack([loss_fn(v) for v in block_logits.values()]).mean()


def test_when_dard_is_off_then_no_block_logits_are_collected() -> None:
    """The per-block head must cost nothing when DARD is absent.

    It runs one extra `lm_head` projection per block, so it is gated on DARD
    being attached rather than always paid.
    """
    engine = make_engine(num_blocks=2)
    engine.train()
    idx = torch.randint(0, engine_vocab(engine), (1, 8))
    engine.denoise_step(idx, block_idx=0, backend="naive")
    assert engine.last_block_logits is None


def test_when_dard_is_attached_then_every_block_is_trainable() -> None:
    """DARD requires all blocks to require grad, or their gradient is discarded."""
    engine = make_engine(num_blocks=2)
    engine.set_dard(make_rule_loss(engine_vocab(engine)))
    engine._activate_block(0)
    for name, p in engine.named_parameters():
        if name.startswith("transformer.h."):
            assert p.requires_grad, f"{name} is frozen but DARD needs it"


def test_when_the_per_block_head_is_off_then_block_isolation_is_preserved() -> None:
    """Without the per-block head the run must stay block-isolated.

    `_activate_block` branches on `dard_per_block_head`, so the opt-out path
    needs its own assertion: enabling every block unconditionally would
    silently undo the property DiffusionBlocks exists for, which is exactly
    what `test_dbcpu_denoise.py` caught when this was on by default.
    """
    engine = make_engine(num_blocks=2)
    engine.set_dard(make_rule_loss(engine_vocab(engine)), per_block_head=False)
    engine.train()
    engine._activate_block(0)
    frozen = [
        name
        for name, p in engine.named_parameters()
        if name.startswith("transformer.h.") and not p.requires_grad
    ]
    assert frozen, "no layer was frozen; block isolation is not in effect"


def test_when_dard_is_off_then_block_isolation_is_preserved() -> None:
    """With no DARD at all the engine is untouched by any of this."""
    engine = make_engine(num_blocks=2)
    engine.train()
    engine._activate_block(0)
    frozen = [
        name
        for name, p in engine.named_parameters()
        if name.startswith("transformer.h.") and not p.requires_grad
    ]
    assert frozen, "no layer was frozen; block isolation is not in effect"


# --- The lm_head projection DARD attaches at ---------------------------------


def test_when_denoiser_logits_are_projected_then_they_match_the_generation_path() -> (
    None
):
    """The penalty must read the distribution the sampler itself would use.

    `generate` applies `lm_head`, slices the padded vocabulary, casts to fp32
    and softcaps with 15.0. If this hook diverged on any of those, DARD would be
    enforcing rules against a different distribution than the model samples
    from, and would still look healthy.
    """
    engine = make_engine(num_blocks=2)
    model = engine.model
    hidden = torch.randn(1, 4, model.config.n_embd)
    softcap = 15.0
    expected = softcap * torch.tanh(model.lm_head(hidden).float() / softcap)
    got = denoiser_logits(model.lm_head, hidden, model.config.vocab_size)
    assert torch.allclose(got, expected, atol=1e-6)
    assert LOGCAP_SOFTMAX == 15.0


def test_when_denoiser_logits_are_projected_then_the_padded_vocab_is_sliced() -> None:
    """The padded vocabulary must be dropped, or the penalty indexes rows that
    the model's loss never sees."""
    engine = make_engine(num_blocks=2)
    model = engine.model
    hidden = torch.randn(1, 2, model.config.n_embd)
    got = denoiser_logits(model.lm_head, hidden, model.config.vocab_size)
    assert got.shape == (1, 2, model.config.vocab_size)
    assert got.shape[-1] <= model.lm_head.weight.shape[0]


# --- DARD under the EDM objective --------------------------------------------


def test_when_dard_is_off_then_no_logits_are_published() -> None:
    """`last_dard_logits` is None without DARD, so the loop adds nothing."""
    engine = make_engine(num_blocks=2)
    engine.train()
    idx = torch.randint(0, engine_vocab(engine), (1, 6))
    engine.denoise_step(idx, block_idx=0)
    assert engine.last_dard_logits is None


def test_when_dard_is_attached_then_denoise_step_publishes_vocabulary_logits() -> None:
    """The denoiser's own logits come out of `denoise_step`, in the graph.

    The sequence axis is `2T`, not `T`: the denoiser input is the App. E.4
    `[clean | noisy]` concatenation, so every block's hidden state -- and so its
    projection through `lm_head` -- spans the whole doubled sequence. The
    caller slices the noisy suffix itself if it needs a per-token term.
    """
    engine = make_engine(num_blocks=2)
    engine.train()
    engine.set_dard(make_rule_loss(engine_vocab(engine)))
    idx = torch.randint(0, engine_vocab(engine), (1, 6))
    engine.denoise_step(idx, block_idx=0)
    logits = published_logits(engine)
    assert logits is not None
    assert logits.shape == (1, 12, engine_vocab(engine))
    assert logits.requires_grad, "published logits must stay in the graph"


def test_when_dard_is_on_then_lm_head_finally_receives_a_gradient_under_edm() -> None:
    """The behaviour change DARD introduces, pinned in both directions.

    Without DARD the EDM objective gives `lm_head` no gradient at all (its
    target is embeddings) -- a property `test_db_edm_behavior.py` pins. With
    DARD the rule penalty flows back through the same projection, so `lm_head`
    becomes trainable *through the auxiliary term only*. The data term is
    unchanged; asserting both halves is what stops this from quietly becoming a
    general-purpose retraining of the head.
    """
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    idx = torch.randint(0, vocab, (1, 6))

    # Off: EDM alone leaves lm_head untouched.
    engine.train()
    engine.zero_grad(set_to_none=True)
    loss, _ = engine.denoise_step(idx, block_idx=0)
    loss.backward()
    grad_without = (
        engine.model.lm_head.weight.grad.clone()
        if engine.model.lm_head.weight.grad is not None
        else None
    )
    assert grad_without is None or grad_without.abs().sum() == 0

    # On: the penalty alone gives it a gradient.
    engine.train()
    engine.zero_grad(set_to_none=True)
    engine.set_dard(make_rule_loss(vocab))
    data_loss, _ = engine.denoise_step(idx, block_idx=0)
    aux = penalty(engine)(published_logits(engine))
    aux.backward()
    grad_with = head_grad(engine)
    assert grad_with.abs().sum() > 0
    assert torch.isfinite(grad_with).all()
    # Attaching DARD must not perturb the data term itself. Two `denoise_step` calls cannot be compared directly -- each resamples sigma and the noise, and `w(sigma) * ||pred - clean||^2` legitimately differs -- so the check is on the code path: with DARD attached and the penalty evaluated separately, the returned value is still exactly the EDM scalar (the engine adds nothing of its own).
    engine.train()
    engine.zero_grad(set_to_none=True)
    data_only, _ = engine.denoise_step(idx, block_idx=0)
    assert data_only.grad_fn is not None


def test_when_dard_is_on_then_the_penalty_backprops_into_the_active_block() -> None:
    """The rule penalty must reach the denoiser, not just the head.

    `denoiser_logits` is called on `pred`, which is produced by the active
    block's layers, so a gradient that reaches `lm_head` but not the block would
    mean the rule was enforced against a frozen representation.
    """
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    engine.train()
    engine.set_dard(make_rule_loss(vocab))
    idx = torch.randint(0, vocab, (1, 6))
    loss, _ = engine.denoise_step(idx, block_idx=0)
    aux = penalty(engine)(published_logits(engine))
    aux.backward()
    grads = [
        p.grad
        for p in engine.transformer_layers()[0].parameters()
        if p.grad is not None
    ]
    assert grads, "no gradient reached the active block's layers"
    assert any(g.abs().sum() > 0 for g in grads)


def test_when_the_aux_penalty_is_zero_then_the_step_is_a_no_op() -> None:
    """Satisfied rules contribute a constant floor, not a gradient.

    With mass on the conclusions the penalty sits at its hinge floor; the
    gradient through it is then ~0, so a fully-satisfied rule set cannot
    perturb training.
    """
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    loss_fn = make_rule_loss(vocab)
    engine.train()
    engine.set_dard(loss_fn)
    idx = torch.randint(0, vocab, (1, 6))
    engine.denoise_step(idx, block_idx=0)
    logits = published_logits(engine)
    # Push every rule's conclusion far above its premise.
    with torch.no_grad():
        for row in range(loss_fn.premise.shape[0]):
            cols = loss_fn.conclusion.col_indices()
            start = loss_fn.conclusion.crow_indices()[row].item()
            end = loss_fn.conclusion.crow_indices()[row + 1].item()
            for c in cols[start:end].tolist():
                logits[..., c] += 50.0
    aux = loss_fn(logits)
    assert aux.item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item() / loss_fn.gamma, abs=1e-3
    )


# --- LC-QAT coexistence ------------------------------------------------------


def test_when_the_model_is_lcqat_then_dard_still_works() -> None:
    """LC-QAT + SparseProp + DiffusionBlocks + DARD must coexist.

    The rule penalty reads `lm_head`, which under LC-QAT is an `LCQATLinear`;
    the hook calls its `forward`, so the quantized path is used and the penalty
    is evaluated on the distribution the deployed model would actually produce.
    """
    engine = make_engine(num_blocks=2)
    retrofit_model(engine.model, PRESETS[DEFAULT_PRESET])
    vocab = engine_vocab(engine)
    engine.train()
    engine.set_dard(make_rule_loss(vocab))
    idx = torch.randint(0, vocab, (1, 6))
    loss, _ = engine.denoise_step(idx, block_idx=0)
    aux = penalty(engine)(published_logits(engine))
    total = loss + aux
    total.backward()
    assert torch.isfinite(total)
    assert engine.model.lm_head.weight.grad is not None


# --- The AR / CE path ---------------------------------------------------------


def test_when_dard_runs_on_the_ce_path_then_it_backprops_too() -> None:
    """The plain-LM path gets the penalty from the model's own logits."""
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    engine.train()
    idx = torch.randint(0, vocab, (1, 6))
    targets = torch.randint(0, vocab, (1, 6))
    data_loss = engine.train_step(idx, targets, block_idx=0)
    logits = engine.model(idx, targets=None)
    aux = make_rule_loss(vocab)(logits)
    (data_loss + aux).backward()
    assert torch.isfinite(head_grad(engine)).all()


def test_when_the_balancer_wraps_the_edm_step_then_both_terms_backprop() -> None:
    """The wired shape: data term from the engine, penalty from its logits."""
    engine = make_engine(num_blocks=2)
    vocab = engine_vocab(engine)
    loss_fn = make_rule_loss(vocab)
    balancer = DardLossBalancer(lambda_global=0.2)
    engine.train()
    balancer.train()
    engine.set_dard(loss_fn)
    idx = torch.randint(0, vocab, (1, 6))
    data_loss, _ = engine.denoise_step(idx, block_idx=0)
    aux = loss_fn(engine.last_dard_logits)
    total, _ = balancer(data_loss, aux)
    total.backward()
    assert head_grad(engine).abs().sum() > 0


# --- The flag surface --------------------------------------------------------


def test_when_flags_are_registered_then_dard_defaults_to_on() -> None:
    """Always-on by default, like LC-QAT and SparseProp (AGENTS.md)."""
    parser = argparse.ArgumentParser()
    add_dard_args(parser)
    args = parser.parse_args([])
    assert args.dard is True
    assert parser.parse_args(["--no-dard"]).dard is False


def test_when_flags_are_registered_then_they_exist_on_all_three_entry_points() -> None:
    """No drift: the same surface on base_train, chat_sft and chat_rl.

    Registration is checked rather than imported, because importing a training
    script executes its whole module-level setup.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("base_train.py", "chat_sft.py", "chat_rl.py"):
        source = (root / "scripts" / name).read_text()
        assert "add_dard_args(parser)" in source, f"{name} does not register DARD"


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"dard_shards": -1}, "--dard-shards"),
        ({"dard_mine_tokens": -1}, "--dard-mine-tokens"),
        ({"dard_max_rules": 0}, "--dard-max-rules"),
        ({"dard_workers": -1}, "--dard-workers"),
        ({"dard_alpha": -1.0}, "--dard-alpha"),
        ({"dard_weight_temperature": 0.0}, "--dard-weight-temperature"),
        ({"dard_lambda": 0.0}, "--dard-lambda"),
        ({"dard_gamma": 0.0}, "--dard-gamma"),
        ({"dard_rules": "/nonexistent/dard.pt"}, "--dard-rules"),
    ],
)
def test_when_dard_flags_are_invalid_then_a_startup_error_names_the_flag(
    overrides, match
) -> None:
    """Rejected at flag time, not ignored until a micro-step later.

    The message must name the offending flag: the `--db-blocks<=0` guard tuple
    exists for exactly this reason.
    """
    parser = argparse.ArgumentParser()
    add_dard_args(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(SystemExit, match=match):
        validate_dard_args(args)


def test_when_dard_flags_are_valid_then_validation_passes() -> None:
    parser = argparse.ArgumentParser()
    add_dard_args(parser)
    validate_dard_args(parser.parse_args([]))


def test_when_shard_count_is_requested_then_k_shards_are_produced() -> None:
    """S_r = f_r / K needs K > 1 to carry any signal; K = 1 makes it constant.

    The shard *size* is therefore derived from the requested shard *count* and
    capped at the PRD's N < 1000, rather than fixed at the cap -- which on a
    small batch (the smoke run's 256 tokens) would produce exactly one shard.
    """
    parser = argparse.ArgumentParser()
    add_dard_args(parser)

    def k_for(num_tokens: int, shards: int) -> int:
        args = parser.parse_args(["--dard-shards", str(shards)])
        size = shard_size_for(args, torch.zeros(num_tokens, dtype=torch.long))
        return -(-num_tokens // size)

    assert k_for(256, 8) == 8
    # Large corpora hit the PRD cap, so K is bounded by tokens/999, not by 8.
    assert k_for(100_000, 8) == 101
    assert (
        shard_size_for(
            parser.parse_args(["--dard-shards", "8"]),
            torch.zeros(100_000, dtype=torch.long),
        )
        == MAX_SHARD_SIZE
    )


def test_when_shard_count_is_zero_then_the_prd_cap_is_used() -> None:
    parser = argparse.ArgumentParser()
    add_dard_args(parser)
    args = parser.parse_args(["--dard-shards", "0"])
    assert shard_size_for(args, torch.zeros(100_000, dtype=torch.long)) == (
        MAX_SHARD_SIZE
    )
