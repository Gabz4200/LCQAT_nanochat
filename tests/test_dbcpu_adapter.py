"""Slice 4: noise-conditioned adapter - sinusoidal embedding, SiLU MLP, AdaLN."""

import pytest
import torch
import torch.nn.functional as F

from nanochat.diffusion_blocks import (
    NoiseConditionedBlockAdapter,
    sinusoidal_noise_embedding,
)


def test_when_log_sigma_zero_then_embedding_is_zero_one_literal() -> None:
    e = sinusoidal_noise_embedding(torch.tensor([0.0]), dim=4)
    assert e.flatten().tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_when_adapter_at_init_then_identity_up_to_rms_norm() -> None:
    torch.manual_seed(0)
    adapter = NoiseConditionedBlockAdapter(n_embd=16, cond_dim=8)
    x = torch.randn(2, 8, 16)
    out = adapter(x, torch.tensor([0.5, 2.0]))
    assert out.shape == x.shape
    assert torch.equal(out, F.rms_norm(x, (16,)))


def test_when_adapter_conditioned_then_sigma_changes_output_and_gets_grad() -> None:
    torch.manual_seed(0)
    adapter = NoiseConditionedBlockAdapter(n_embd=16, cond_dim=8)
    # Break zero-init symmetry so conditioning has an effect.
    with torch.no_grad():
        adapter.out.weight.fill_(0.01)
    x = torch.randn(2, 8, 16)
    y1 = adapter(x, torch.tensor([0.5, 0.5]))
    y2 = adapter(x, torch.tensor([4.0, 4.0]))
    assert not torch.equal(y1, y2)
    y1.sum().backward()
    assert adapter.out.weight.grad is not None
