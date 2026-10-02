"""Per-layer AdaLN conditioning (W1.4).

DiffusionBlocks Step 3 conditions the block on sigma from the *inside*. Before
W1.4, `NoiseConditionedBlockAdapter` returned `rms_norm(x) * (1+gamma) + beta`
applied to the block's *input stream*, which normalized the residual stream before
the block's own pre-norm and discarded the residual identity that Section 2.2
of the paper depends on ("residual connections naturally correspond to updates
in a dynamical system"). Conditioning was also per-block-group, so every layer
in a block saw an identical modulation.

The default path must be untouched: `Block(cond=None)` has to be bit-identical
to the pre-W1.4 block, or every checkpoint and caller breaks.
"""

import torch

from nanochat.training.diffusion_blocks import NoiseConditionedBlockAdapter
from tests.conftest import build_active_tiny_gpt


def _block_forward(cond=None):
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    block = model.transformer.h[0]
    x = torch.randn(2, 8, model.config.n_embd)
    cos_sin = model._precompute_rotary_embeddings(
        8, model.config.n_embd // model.config.n_head
    )
    with torch.no_grad():
        return block(x, None, cos_sin, (8, 0), None, cond=cond)


def test_when_cond_is_none_then_the_block_is_the_plain_nanochat_block():
    """The guard that keeps every existing caller and checkpoint valid."""
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    block = model.transformer.h[0]
    x = torch.randn(2, 8, model.config.n_embd)
    cos_sin = model._precompute_rotary_embeddings(
        8, model.config.n_embd // model.config.n_head
    )
    with torch.no_grad():
        via_kwarg = block(x, None, cos_sin, (8, 0), None, cond=None)
        via_positional = block(x, None, cos_sin, (8, 0), None)
    assert torch.equal(via_kwarg, via_positional)


def test_when_cond_is_none_then_it_matches_an_unconditioned_reference():
    """Recompute the block by hand: pre-norm, attn residual, pre-norm, mlp residual."""
    from nanochat.models.backbone import norm

    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    block = model.transformer.h[0]
    x = torch.randn(2, 8, model.config.n_embd)
    cos_sin = model._precompute_rotary_embeddings(
        8, model.config.n_embd // model.config.n_head
    )
    with torch.no_grad():
        got = block(x, None, cos_sin, (8, 0), None, cond=None)
        h = norm(x)
        h = x + block.attn(h, None, cos_sin, (8, 0), None)
        want = h + block.mlp(norm(h))
    assert torch.allclose(got, want, atol=1e-6)


def test_when_cond_is_provided_then_the_output_changes():
    torch.manual_seed(0)
    base = _block_forward(cond=None)
    n_embd = 256
    cond = (
        torch.full((2, 1, n_embd), 0.5),
        torch.full((2, 1, n_embd), 0.1),
    )
    modulated = _block_forward(cond=cond)
    assert not torch.allclose(base, modulated)


def test_when_cond_shapes_then_broadcast_over_the_sequence_axis():
    """gamma/beta are (B, 1, n_embd): batch-varying, sequence-shared."""
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    block = model.transformer.h[0]
    x = torch.randn(2, 8, model.config.n_embd)
    cos_sin = model._precompute_rotary_embeddings(
        8, model.config.n_embd // model.config.n_head
    )
    gamma = torch.randn(2, 1, model.config.n_embd)
    beta = torch.randn(2, 1, model.config.n_embd)
    with torch.no_grad():
        out = block(x, None, cos_sin, (8, 0), None, cond=(gamma, beta))
    assert out.shape == x.shape
    # A (B, T, C) broadcast of the same cond must give the same result, since
    # the modulation is sequence-independent by construction.
    gamma_t = gamma.expand(2, 8, model.config.n_embd)
    beta_t = beta.expand(2, 8, model.config.n_embd)
    with torch.no_grad():
        out_t = block(x, None, cos_sin, (8, 0), None, cond=(gamma_t, beta_t))
    assert torch.allclose(out, out_t, atol=1e-5)


def test_when_adapter_at_init_then_conditioning_is_an_exact_no_op():
    """Zero-init of the adapter's output layer: gamma=0, beta=0 exactly.

    That is what makes `h * (1 + 0) + 0 == h`, so a fresh model is identical to
    the unconditioned one and training starts from a clean identity.
    """
    torch.manual_seed(0)
    adapter = NoiseConditionedBlockAdapter(n_embd=256, cond_dim=8, n_layers=2)
    x = torch.randn(2, 8, 256)
    conds = adapter(x, torch.tensor([0.1, 5.0]))
    assert len(conds) == 2
    for gamma, beta in conds:
        assert gamma.shape == (2, 1, 256)
        assert torch.equal(gamma, torch.zeros_like(gamma))
        assert torch.equal(beta, torch.zeros_like(beta))


def test_when_adapter_trained_then_each_layer_gets_its_own_modulation():
    """A per-block modulation could not express this: gamma_i must differ by i."""
    torch.manual_seed(1)
    adapter = NoiseConditionedBlockAdapter(n_embd=256, cond_dim=8, n_layers=4)
    with torch.no_grad():
        adapter.mlp[-1].weight.normal_(std=0.05)
    conds = adapter(torch.randn(2, 8, 256), torch.tensor([1.0, 1.0]))
    gammas = [g.reshape(-1) for g, _ in conds]
    for i in range(len(gammas) - 1):
        assert not torch.equal(gammas[i], gammas[i + 1])


def test_when_adapter_emits_pairs_then_n_layers_pairs_are_returned():
    adapter = NoiseConditionedBlockAdapter(n_embd=64, cond_dim=8, n_layers=3)
    assert adapter.n_layers == 3
    conds = adapter(torch.randn(2, 8, 64), torch.tensor([1.0, 1.0]))
    assert len(conds) == 3
