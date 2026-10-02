"""Slice 6: packed LM batches, B=1 loss equivalence, 1/B grad-param ratio."""

import torch

from nanochat.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
    packed_lm_batch,
)
from tests.conftest import build_active_tiny_gpt


def test_when_packed_batch_then_shifted_targets_and_exact_length() -> None:
    idx, targets = packed_lm_batch(
        [1, 2, 3, 4, 5, 6, 7, 8], seq_len=4, batch_size=2, step=0
    )
    assert idx.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert targets.tolist() == [[2, 3, 4, 5], [6, 7, 8, 1]]


def test_when_single_block_then_train_step_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=1))
    idx = torch.randint(0, 128, (2, 16))
    assert engine.train_step(idx, idx, block_idx=0).item() == model(idx, idx).item()


def test_when_denoise_step_then_transformer_grads_are_confined_to_one_block() -> None:
    """The 1/B gradient-memory property, stated over the layers that dominate it.

    Shared parameters (the embedding table, `lm_head`, scalars) legitimately train
    on every step -- the embedding in particular, because the denoiser is
    conditioned on clean token embeddings. What must be confined is the
    transformer-layer parameter count, which is where the memory actually is.
    """
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
    idx = torch.randint(0, 128, (2, 16))
    loss, _ = engine.denoise_step(idx, block_idx=0)
    loss.backward()

    layer_grads = sum(
        p.numel() for p in model.transformer.h[0].parameters() if p.grad is not None
    )
    other_layer_grads = sum(
        p.numel() for p in model.transformer.h[1].parameters() if p.grad is not None
    )
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())
    assert layer_grads == block_params, "the active block's layers all train"
    assert other_layer_grads == 0, "the inactive block's layers do not"

    # And the shared embedding trains too (see diffusion_blocks.denoise_step).
    assert model.transformer.wte.weight.grad is not None
