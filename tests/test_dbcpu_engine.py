"""Slice 2: single-block train_step isolates gradients to the active block."""

import torch

from nanochat.diffusion_blocks import DiffusionBlockEngine, EquiProbabilityPartitioner
from tests.conftest import build_active_tiny_gpt


def test_when_train_step_then_only_active_block_gets_grad() -> None:
    torch.manual_seed(0)
    model = build_active_tiny_gpt()  # n_layer=2, non-zero blocks
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=2), dtype=torch.float32
    )
    idx = torch.randint(0, 128, (2, 16))
    loss = engine.train_step(idx, idx, block_idx=0)
    assert torch.isfinite(loss).all()
    loss.backward()

    active = [p.grad is not None for p in model.transformer.h[0].parameters()]
    frozen = [p.grad is not None for p in model.transformer.h[1].parameters()]
    assert any(active)
    assert not any(frozen)


def test_when_train_step_then_all_blocks_cover_all_layers() -> None:
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=2), dtype=torch.float32
    )
    assert engine.block_layers() == [[0], [1]]
