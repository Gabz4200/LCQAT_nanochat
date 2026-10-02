"""Tests for DiffusionBlockEngine Euler diffusion generation and block-diagonal attention masking."""

import torch

from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
    block_diagonal_mask,
)
from tests.conftest import build_active_tiny_gpt


def test_block_diagonal_mask_structure():
    """Verify block diagonal mask preserves within-doc causal attention and blocks cross-doc."""
    seq_lens = [3, 2, 4]
    seq_len = 9
    mask = block_diagonal_mask(seq_lens, seq_len, dtype=torch.bool)
    assert mask.shape == (1, 1, 9, 9)
    m = mask[0, 0]

    # Check causal property: no future attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert not m[i, j].item(), (
                f"Future position ({i}, {j}) must not be attended to"
            )

    # Doc 0 is positions 0, 1, 2
    # Position 2 should attend to 0, 1, 2
    assert m[2, 0].item() and m[2, 1].item() and m[2, 2].item()

    # Doc 1 is positions 3, 4
    # Position 3 should attend to 3, but NOT 0, 1, 2
    assert m[3, 3].item()
    assert not m[3, 0].item()
    assert not m[3, 1].item()
    assert not m[3, 2].item()

    # Doc 2 is positions 5, 6, 7, 8
    # Position 5 should attend to 5, but NOT 4
    assert m[5, 5].item()
    assert not m[5, 4].item()


def test_block_diagonal_mask_float_inf():
    """Verify float mask produces 0.0 for allowed and -inf for blocked."""
    seq_lens = [2, 2]
    mask = block_diagonal_mask(seq_lens, 4, dtype=torch.float32)
    m = mask[0, 0]
    assert m[0, 0].item() == 0.0
    assert m[1, 0].item() == 0.0
    assert m[2, 0].item() == float("-inf")
    assert m[0, 1].item() == float("-inf")


def test_diffusion_generate_tokens():
    """Euler diffusion generation produces valid tokens of expected length."""
    model = build_active_tiny_gpt()
    partitioner = EquiProbabilityPartitioner(num_blocks=2)
    engine = DiffusionBlockEngine(model, partitioner)

    # Empty prompt generation
    tokens = engine.generate(idx=None, max_new_tokens=5, seed=42)
    assert len(tokens) == 5
    assert all(isinstance(t, int) and 0 <= t < model.config.vocab_size for t in tokens)

    # Prompt conditioning generation
    prompt = [10, 20]
    gen_tokens = engine.generate(idx=prompt, max_new_tokens=4, seed=42)
    assert len(gen_tokens) == len(prompt) + 4
    assert gen_tokens[: len(prompt)] == prompt
    assert all(
        isinstance(t, int) and 0 <= t < model.config.vocab_size for t in gen_tokens
    )


def test_diffusion_engine_train_step_with_mask():
    """train_step accepts block-diagonal attention mask and runs clean backward."""
    model = build_active_tiny_gpt()
    partitioner = EquiProbabilityPartitioner(num_blocks=2)
    engine = DiffusionBlockEngine(model, partitioner)

    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    mask = block_diagonal_mask([4, 4], 8)

    loss = engine.train_step(idx, targets, block_idx=0, attn_mask=mask)
    assert torch.isfinite(loss)
    loss.backward()

    # Block 0 has layer 0
    for p in model.transformer.h[0].parameters():
        assert p.grad is not None
    # Block 1 (layer 1) must NOT have grads
    for p in model.transformer.h[1].parameters():
        assert p.grad is None


def test_diffusion_engine_logprobs():
    """logprobs computes unreduced token loss for RL with isolated block grads."""
    model = build_active_tiny_gpt()
    partitioner = EquiProbabilityPartitioner(num_blocks=2)
    engine = DiffusionBlockEngine(model, partitioner)

    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))

    loss_unreduced = engine.logprobs(idx, targets, block_idx=1)
    assert loss_unreduced.shape == (2, 6)
    assert torch.isfinite(loss_unreduced).all()
