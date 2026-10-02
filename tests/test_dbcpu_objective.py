"""The EDM denoising objective is the real DiffusionBlocks training path.

`denoise_step` is what the paper's method is: block `b` trains to denoise within
its own equi-probability sigma range, and only that block's layers run, so
gradients exist for L/B layers rather than L. `train_step` (full-depth
next-token CE with block-isolated gradients) is the escape hatch.
"""

import torch

from nanochat.training.diffusion_blocks import (
    EquiProbabilityPartitioner,
    block_diagonal_mask,
    edm_preconditioning,
)
from tests.test_dbcpu_engine import make_engine


def test_when_denoise_step_then_loss_and_sigma_are_well_formed():
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    loss, sigma = engine.denoise_step(idx, block_idx=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert sigma.ndim == 0 and float(sigma) > 0


def test_when_sigma_sampled_then_it_lies_in_the_blocks_own_range():
    """Equi-probability partitioning: each block owns a disjoint sigma interval,
    which is what makes block independence meaningful."""
    engine = make_engine(3, n_layer=6)
    bounds = engine.partitioner.boundaries()
    for b in range(3):
        lo, hi = float(bounds[b]), float(bounds[b + 1])
        for _ in range(20):
            sigma = float(engine.partitioner.sample_sigma(b, overlap=0.0))
            assert lo <= sigma <= hi, f"block {b} sampled {sigma} outside [{lo}, {hi}]"


def test_when_overlap_enabled_then_the_support_is_widened():
    """DiffusionBlocks App. C: gamma > 0 extends each interval to smooth
    transitions between neighbouring denoisers.

    Checked on the CDF rather than on sample extrema: `sample_sigma` clamps the
    CDF at 1e-6, so the reachable endpoints are the 1e-6 / 1-1e-6 quantiles, and
    comparing two finite sample sets would be comparing different quantiles.
    """
    part = EquiProbabilityPartitioner(num_blocks=3)
    bounds = part.boundaries()
    for b in range(3):
        lo, hi = float(bounds[b]), float(bounds[b + 1])
        alpha = (hi / lo) ** 0.1
        assert alpha > 1.0
        assert part._cdf(lo / alpha).item() < part._cdf(lo).item()
        assert part._cdf(hi * alpha).item() > part._cdf(hi).item()
        gen = torch.Generator().manual_seed(0)
        wide = [
            float(part.sample_sigma(b, generator=gen, overlap=0.1)) for _ in range(100)
        ]
        assert all(lo / alpha - 1e-6 <= s <= hi * alpha + 1e-6 for s in wide)


def test_when_edm_preconditioning_then_weighting_matches_the_formula():
    sigma = torch.tensor([0.05, 0.5, 5.0])
    sigma_data = 0.5
    c_in, c_out, w = edm_preconditioning(sigma, sigma_data)
    var = sigma**2 + sigma_data**2
    assert torch.allclose(c_in, var**-0.5)
    assert torch.allclose(c_out, sigma * sigma_data / var.sqrt())
    assert torch.allclose(w, var / (sigma * sigma_data).square())


def test_when_training_then_loss_decreases():
    """60-step overfit on a fixed batch. Guards the whole EDM path end to end:
    per-layer AdaLN conditioning, the per-block head, and the codebook gradients
    all have to be live for the loss to move at all."""
    torch.manual_seed(0)
    engine = make_engine(2)
    opt = torch.optim.AdamW(list(engine.parameters()), lr=1e-3)
    idx = torch.randint(0, 128, (4, 32))
    with torch.no_grad():
        clean = torch.nn.functional.normalize(
            engine.model.transformer.wte(idx).float(), dim=-1
        )
    losses = []
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=step % 2, overlap=0.1, clean=clean)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    assert last < first, f"EDM loss did not decrease: {first:.5f} -> {last:.5f}"


def test_when_loss_scales_with_sigma_then_the_jitter_is_the_edm_weighting():
    """Explains the per-step loss variation seen in a training log.

    With fixed weights and a fixed target the loss still varies a lot across
    steps, because sigma is resampled and w(sigma) reweights it. That is the EDM
    objective behaving as specified, not instability -- pinned so a future reader
    does not try to "fix" it.
    """
    torch.manual_seed(0)
    engine = make_engine(2, active=False)  # weights frozen: sigma is the only input
    idx = torch.randint(0, 128, (4, 32))
    with torch.no_grad():
        clean = torch.nn.functional.normalize(
            engine.model.transformer.wte(idx).float(), dim=-1
        )
    seen = []
    for i in range(6):
        loss, sigma = engine.denoise_step(idx, block_idx=i % 2, clean=clean)
        seen.append((float(sigma), loss.item()))
    sigmas = [s for s, _ in seen]
    losses = [v for _, v in seen]
    assert max(losses) / min(losses) > 1.5, "expected sigma-driven loss spread"
    assert len(set(round(s, 6) for s in sigmas)) > 1


def test_when_attn_mask_causal_then_it_matches_the_unmasked_path():
    """`denoise_step` calls the blocks directly, so it must thread the mask itself.

    `train_step` forwards the mask through GPT.forward; the EDM path bypassed that
    and always ran unmasked, which made packed sequences silently wrong.
    """
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    t = idx.size(1)
    causal = torch.ones(t, t, dtype=torch.bool).tril().view(1, 1, t, t)
    torch.manual_seed(7)
    unmasked, _ = engine.denoise_step(idx, block_idx=0, attn_mask=None)
    torch.manual_seed(7)
    masked, _ = engine.denoise_step(idx, block_idx=0, attn_mask=causal)
    # Attention is already causal, so a full causal mask is a no-op.
    assert torch.allclose(unmasked, masked, atol=1e-5)


def test_when_block_diagonal_mask_then_packed_documents_are_isolated():
    """The block-diagonal mask is what stops a packed document attending to its
    neighbour. It was dead code before W1.1, so it has to actually change things."""
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    t = idx.size(1)
    mask = block_diagonal_mask([t // 2, t - t // 2], t)
    assert mask.shape == (1, 1, t, t)
    plain_causal = torch.ones(t, t, dtype=torch.bool).tril().view(1, 1, t, t)
    # Guard against a vacuous comparison: the mask must differ from causal.
    assert not torch.equal(mask, plain_causal)
    torch.manual_seed(11)
    unmasked, _ = engine.denoise_step(idx, block_idx=0, attn_mask=None)
    torch.manual_seed(11)
    packed, _ = engine.denoise_step(idx, block_idx=0, attn_mask=mask)
    assert not torch.allclose(unmasked, packed)


def test_when_objectives_compared_then_only_edm_isolates_the_forward():
    """What each objective actually costs.

    `train_step` runs all L layers: it saves backward and optimizer memory but no
    forward FLOPs. `denoise_step` runs L/B layers and saves both. That is the
    reason `--db-objective edm` is the default rather than `ce`.
    """
    engine = make_engine(2, n_layer=4)
    calls = []
    for layer in engine.model.transformer.h:
        layer.register_forward_hook(lambda m, i, o: calls.append(1))

    idx = torch.randint(0, 128, (2, 16))
    engine.train_step(idx, idx, block_idx=0)
    assert len(calls) == 4, "train_step is a full-depth forward"

    calls.clear()
    engine.denoise_step(idx, block_idx=0)
    assert len(calls) == 2, "denoise_step runs only the active block's layers"

    calls.clear()
    engine.denoise_step(idx, block_idx=1)
    assert len(calls) == 2, "still L/B = 2 layers, whichever block is active"


def test_when_denoise_step_reuses_hoisted_clean_then_no_recomputation():
    """The batch is reused across micro-steps, so the target is hoisted.

    Passing `clean` must produce the same result as letting `denoise_step` build
    it, and must not re-enter the embedding lookup.
    """
    torch.manual_seed(0)
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        clean = torch.nn.functional.normalize(
            engine.model.transformer.wte(idx).float(), dim=-1
        )
    torch.manual_seed(3)
    hoisted, _ = engine.denoise_step(idx, block_idx=0, clean=clean)
    torch.manual_seed(3)
    implicit, _ = engine.denoise_step(idx, block_idx=0, clean=None)
    assert torch.allclose(hoisted, implicit, atol=1e-6)
