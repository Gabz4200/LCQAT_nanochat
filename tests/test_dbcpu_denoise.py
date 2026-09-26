"""Slice 5: EDM denoising step - truncated sigma sampling, c_noise, w-weighted L2."""

import math

import pytest
import torch

from nanochat.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
    c_noise,
)
from tests.conftest import build_active_tiny_gpt


def test_when_c_noise_then_edm_log_scaling_literals() -> None:
    assert c_noise(torch.tensor(1.0)).item() == pytest.approx(0.0)
    assert c_noise(torch.tensor(math.e**4)).item() == pytest.approx(1.0)


def test_when_sample_sigma_then_inside_block_interval() -> None:
    torch.manual_seed(0)
    p = EquiProbabilityPartitioner(num_blocks=4)
    bounds = p.noise_boundaries
    for b in range(4):
        s = p.sample_sigma(b, generator=torch.Generator().manual_seed(b))
        assert bounds[b].item() <= s.item() <= bounds[b + 1].item()


def test_when_denoise_step_then_finite_loss_and_grads_only_in_active_block() -> None:
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
    idx = torch.randint(0, 128, (2, 16))
    loss, sigma = engine.denoise_step(idx, block_idx=1)
    assert torch.isfinite(loss).all()
    bounds = engine.partitioner.noise_boundaries
    assert bounds[1].item() <= sigma.item() <= bounds[2].item()
    loss.backward()
    assert all(p.grad is None for p in model.transformer.h[0].parameters())
    assert any(p.grad is not None for p in model.transformer.h[1].parameters())


def test_when_denoise_step_twice_then_different_sigmas() -> None:
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
    idx = torch.randint(0, 128, (2, 16))
    _, s1 = engine.denoise_step(idx, block_idx=0)
    _, s2 = engine.denoise_step(idx, block_idx=0)
    assert s1.item() != s2.item()
