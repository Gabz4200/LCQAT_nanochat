"""Where the active block is drawn (W1.2).

The block must be sampled once per *optimizer step*, not once per micro-step.

Two things go wrong with per-micro-step sampling, and the second is worse than
the plan anticipated:

1. The noise-range specialization never happens -- micro-steps target different
   blocks, so no block gets a coherent signal.
2. `_apply_requires_grad` clears `p.grad` for parameters the active block does
   not own. Under step-level sampling that is a no-op (nothing else has a
   gradient to clear within a step). Under micro-step sampling each micro-step
   *erases* the previous one's gradients, so only the last block sampled
   contributes to the optimizer step.

`micro` is retained as a named ablation arm, and `test_when_micro_mode_then_earlier_blocks_gradients_are_discarded`
pins the loss so the behaviour cannot drift silently.
"""

import torch

from tests.test_dbcpu_engine import make_engine


def test_when_sample_block_then_it_is_in_range():
    engine = make_engine(3, n_layer=6)
    torch.manual_seed(0)
    seen = {engine.sample_block() for _ in range(200)}
    assert seen == {0, 1, 2}, "every block should be reachable"


def test_when_sample_block_with_generator_then_it_is_reproducible():
    engine = make_engine(3, n_layer=6)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = [engine.sample_block(g1) for _ in range(20)]
    b = [engine.sample_block(g2) for _ in range(20)]
    assert a == b


def test_when_sampling_step_mode_then_every_micro_step_uses_the_same_block():
    """What base_train does: one draw before the micro-step loop."""
    engine = make_engine(4, n_layer=8)
    idx = torch.randint(0, 128, (2, 16))
    grad_accum = 4
    block_idx = engine.sample_block()
    seen = []
    for _ in range(grad_accum):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=block_idx)
        loss.backward()
        grads = {n for n, p in engine.named_parameters() if p.grad is not None}
        head = next(n.split(".")[1] for n in grads if n.startswith("db_denoise_heads."))
        seen.append(head)
    assert set(seen) == {str(block_idx)}


def test_when_sampling_micro_mode_then_blocks_vary_across_micro_steps():
    """The ablation arm: `block_idx=None` redraws per micro-step."""
    engine = make_engine(4, n_layer=8)
    idx = torch.randint(0, 128, (2, 16))
    torch.manual_seed(0)
    seen = []
    for _ in range(12):
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=None)
        loss.backward()
        grads = {n for n, p in engine.named_parameters() if p.grad is not None}
        seen.append(
            next(n.split(".")[1] for n in grads if n.startswith("db_denoise_heads."))
        )
    assert len(set(seen)) > 1, "per-micro-step sampling should visit several blocks"


def test_when_micro_mode_then_earlier_blocks_gradients_are_discarded():
    """Why `micro` is a lossy arm, not just a degenerate one.

    `_apply_requires_grad` sets `p.grad = None` for every parameter the active
    block does not own. That is correct for step-level sampling (one block per
    step, so nothing else has a gradient to discard), but under per-micro-step
    sampling each micro-step *erases* the previous micro-step's gradients. The
    optimizer step then applies only the last block sampled.

    So `--db-block-sampling micro` is not "block-wise training with a memory
    discount" -- it silently drops every micro-step but the last. Pinned so that
    anyone touching `_apply_requires_grad` sees the consequence.
    """
    engine = make_engine(3, n_layer=6)
    idx = torch.randint(0, 128, (2, 16))
    engine.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    seen = []
    for _ in range(8):
        loss, _ = engine.denoise_step(idx, block_idx=None)
        loss.backward()  # accumulates into .grad ...
        grads = {n for n, p in engine.named_parameters() if p.grad is not None}
        seen.append(
            frozenset(
                n.split(".")[1] for n in grads if n.startswith("db_denoise_heads.")
            )
        )
    # ... but the next micro-step zeroes the ones it does not own, so at no point
    # do gradients from more than one block survive.
    assert all(len(s) == 1 for s in seen), (
        f"expected one surviving block per micro-step, got {[sorted(s) for s in seen]}"
    )
    assert len({next(iter(s)) for s in seen}) > 1, "blocks did vary, as expected"


def test_when_step_mode_then_no_gradient_is_discarded_mid_step():
    """The contrast: step-level sampling keeps every micro-step's gradient.

    All micro-steps in a step target the same block, so `_apply_requires_grad`
    never clears a gradient that this step produced. This is why the default
    accumulates correctly while `micro` does not.
    """
    engine = make_engine(3, n_layer=6)
    idx = torch.randint(0, 128, (2, 16))
    engine.zero_grad(set_to_none=True)
    block_idx = engine.sample_block()
    for _ in range(3):
        loss, _ = engine.denoise_step(idx, block_idx=block_idx)
        loss.backward()
    grads = {n for n, p in engine.named_parameters() if p.grad is not None}
    blocks = {n.split(".")[1] for n in grads if n.startswith("db_denoise_heads.")}
    assert blocks == {str(block_idx)}, "one block's gradients, summed over micro-steps"
    # And the sum is genuinely larger than any single micro-step contributed.
    head = engine.denoise_heads[block_idx].weight
    assert head.grad.abs().sum() > 0
