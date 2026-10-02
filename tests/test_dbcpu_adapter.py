"""Slice 4: noise-conditioned adapter - sinusoidal embedding, SiLU MLP, AdaLN.

The adapter emits one `(gamma, beta)` pair *per layer* of its block rather than
one for the whole group: DiffusionBlocks conditions inside the block, so each
layer responds to sigma independently. The output layer is zero-initialized, so a
fresh adapter is an exact no-op (gamma=0, beta=0), which is what makes a
freshly-built model byte-identical to the unconditioned one.
"""

import pytest
import torch

from nanochat.diffusion_blocks import (
    NoiseConditionedBlockAdapter,
    sinusoidal_noise_embedding,
)


def test_when_log_sigma_zero_then_embedding_is_zero_one_literal() -> None:
    e = sinusoidal_noise_embedding(torch.tensor([0.0]), dim=4)
    assert e.flatten().tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_when_adapter_at_init_then_conditioning_is_exact_no_op() -> None:
    torch.manual_seed(0)
    adapter = NoiseConditionedBlockAdapter(n_embd=16, cond_dim=8, n_layers=3)
    x = torch.randn(2, 8, 16)
    conds = adapter(x, torch.tensor([0.5, 2.0]))
    assert len(conds) == 3, "one (gamma, beta) pair per layer of the block"
    for gamma, beta in conds:
        assert gamma.shape == (2, 1, 16), "broadcast over the sequence axis"
        assert beta.shape == (2, 1, 16)
        # Zero-init of the output layer: gamma=0, beta=0 exactly, so applying
        # the conditioning to a pre-norm activation is an exact identity.
        assert torch.equal(gamma, torch.zeros_like(gamma))
        assert torch.equal(beta, torch.zeros_like(beta))


def test_when_adapter_conditioned_then_sigma_changes_each_layer_and_gets_grad() -> None:
    torch.manual_seed(0)
    adapter = NoiseConditionedBlockAdapter(n_embd=16, cond_dim=8, n_layers=3)
    # Break zero-init symmetry so conditioning has an effect.
    # The final linear is mlp[-1] (no self.out alias -- that would share the
    # Parameter object under two paths and trip the optimizer's dup-param check).
    with torch.no_grad():
        adapter.mlp[-1].weight.normal_(std=0.05)
    x = torch.randn(2, 8, 16)
    low = adapter(x, torch.tensor([0.5, 0.5]))
    high = adapter(x, torch.tensor([4.0, 4.0]))
    assert len(low) == len(high) == 3
    # Every layer's conditioning depends on sigma, not just the first.
    for (g_lo, b_lo), (g_hi, b_hi) in zip(low, high, strict=True):
        assert not torch.equal(g_lo, g_hi)
        assert not torch.equal(b_lo, b_hi)
    low[0][0].sum().backward()
    assert adapter.mlp[-1].weight.grad is not None
    assert adapter.mlp[-1].weight.grad.abs().sum() > 0


def test_when_layers_differ_then_conditioning_is_per_layer_distinct() -> None:
    """A per-block modulation could not express this: gamma_i must differ by i."""
    torch.manual_seed(1)
    adapter = NoiseConditionedBlockAdapter(n_embd=16, cond_dim=8, n_layers=4)
    with torch.no_grad():
        adapter.mlp[-1].weight.normal_(std=0.05)
    conds = adapter(torch.randn(2, 8, 16), torch.tensor([1.0, 1.0]))
    gammas = [g.reshape(-1) for g, _ in conds]
    for i in range(len(gammas) - 1):
        assert not torch.equal(gammas[i], gammas[i + 1])
