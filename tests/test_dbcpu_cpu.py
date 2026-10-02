"""Slice 3: CPU backend - thread pin, FP32 AdamW step, sequence packing."""

import torch

from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
    configure_cpu_training,
    cpu_adamw_for,
    pack_sequences,
)
from tests.conftest import build_active_tiny_gpt


def test_when_configure_cpu_then_four_threads() -> None:
    configure_cpu_training()
    assert torch.get_num_threads() == 4


def test_when_cpu_adamw_step_then_active_params_update() -> None:
    torch.manual_seed(1)
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
    opt = cpu_adamw_for(engine, lr=3e-4)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    idx = torch.randint(0, 128, (2, 16))
    before = model.transformer.h[0].attn.c_q.weight.detach().clone()
    opt.zero_grad(set_to_none=True)
    loss = engine.train_step(idx, idx, block_idx=0)
    loss.backward()
    opt.step()
    assert not torch.equal(before, model.transformer.h[0].attn.c_q.weight)


def test_when_pack_sequences_then_concatenates_and_chunks() -> None:
    out = pack_sequences([[1, 2, 3], [4, 5], [6, 7, 8, 9]], seq_len=4)
    assert out == [[1, 2, 3, 4], [5, 6, 7, 8]]
