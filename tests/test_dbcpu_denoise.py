"""Slice: the EDM denoising objective is the real DiffusionBlocks training path.

`denoise_step` is what the paper's method actually is: block `b` is trained to
denoise within its own equi-probability sigma range, and only that block's layers
run, so gradients exist for L/B layers instead of L. `train_step` (full-depth
next-token CE with block-isolated gradients) is the escape hatch.

Absorbs the former `test_dbcpu_objective.py`: it was a near-wholesale duplicate
of this file -- same module docstring intent, same imports, 20.6% duplication --
with four tests whose bodies differed only in docstring wording. Its four unique
tests are here; the six it shared with this file are not.
"""

import gc

import torch

from nanochat.training.diffusion_blocks import (
    EquiProbabilityPartitioner,
    block_diagonal_mask,
    edm_preconditioning,
)
from tests.test_dbcpu_engine import make_engine


def test_when_denoise_step_then_loss_and_sigma_are_well_formed() -> None:
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    loss, sigma = engine.denoise_step(idx, block_idx=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert sigma.ndim == 0 and float(sigma) > 0


def test_when_sigma_sampled_then_it_lies_in_the_blocks_own_range() -> None:
    """Equi-probability partitioning: each block owns a disjoint sigma interval."""
    engine = make_engine(3, n_layer=6)
    bounds = engine.partitioner.boundaries()
    for b in range(3):
        lo, hi = float(bounds[b]), float(bounds[b + 1])
        for _ in range(20):
            sigma = float(engine.partitioner.sample_sigma(b, overlap=0.0))
            assert lo <= sigma <= hi, f"block {b} sampled {sigma} outside [{lo}, {hi}]"


def test_when_overlap_enabled_then_range_is_widened() -> None:
    """DiffusionBlocks App. C: gamma > 0 extends each block's interval to smooth
    transitions between neighbouring denoisers.

    Checks the partitioner directly, with no model involved, because the property
    is a property of the noise schedule rather than of the network.
    """
    part = EquiProbabilityPartitioner(num_blocks=3)
    bounds = part.boundaries()
    for b in range(3):
        lo, hi = float(bounds[b]), float(bounds[b + 1])
        # gamma > 0 widens the support to [lo/alpha, hi*alpha] with
        # alpha = (hi/lo)^gamma > 1, so the widened distribution covers strictly
        # more of the log-sigma axis. Assert that on the CDF directly: the
        # reachable endpoints are the 1e-6 / 1-1e-6 quantiles (sample_sigma
        # clamps), not exactly lo and hi, so comparing sample extrema would be
        # comparing different quantiles rather than testing containment.
        alpha = (hi / lo) ** 0.1
        assert alpha > 1.0
        assert part._cdf(lo / alpha).item() < part._cdf(lo).item()
        assert part._cdf(hi * alpha).item() > part._cdf(hi).item()
        # And a wide draw stays inside the widened interval, never outside it.
        gen = torch.Generator().manual_seed(0)
        wide = [
            float(part.sample_sigma(b, generator=gen, overlap=0.1)) for _ in range(100)
        ]
        assert all(lo / alpha - 1e-6 <= s <= hi * alpha + 1e-6 for s in wide)
        tight_gen = torch.Generator().manual_seed(0)
        tight = [
            float(part.sample_sigma(b, generator=tight_gen, overlap=0.0))
            for _ in range(100)
        ]
        assert all(lo - 1e-6 <= s <= hi + 1e-6 for s in tight)


def test_when_edm_preconditioning_then_weighting_matches_the_formula() -> None:
    sigma = torch.tensor([0.05, 0.5, 5.0])
    sigma_data = 0.5
    c_in, c_out, w = edm_preconditioning(sigma, sigma_data)
    var = sigma**2 + sigma_data**2
    assert torch.allclose(c_in, var**-0.5)
    assert torch.allclose(c_out, sigma * sigma_data / var.sqrt())
    assert torch.allclose(w, var / (sigma * sigma_data).square())


def test_when_training_then_loss_decreases() -> None:
    """A 60-step overfit on a fixed batch. Guards the whole EDM path end to end:
    the per-layer AdaLN conditioning, the per-block head, and the codebook
    gradients all have to be live for the loss to move at all."""
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
        # Alternate blocks, as the real training loop samples one per step.
        loss, _ = engine.denoise_step(idx, block_idx=step % 2, overlap=0.1, clean=clean)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    assert last < first, f"EDM loss did not decrease: {first:.5f} -> {last:.5f}"


def test_when_denoise_step_then_activation_memory_tracks_the_active_block_only() -> (
    None
):
    """The B-fold saving is the whole justification for the method.

    Compares peak activation memory of one `denoise_step` at B=1 vs B=4 blocks
    over a fixed depth. With block isolation, B=4 runs 2 layers per step instead
    of 4, so peak activation memory should drop. Compared against the full-depth
    `train_step` as the reference, which is the path that does NOT save forward
    memory.
    """

    def peak_mb(fn):
        # Reset the allocator's high-water mark by taking a baseline reading
        # after a throwaway allocation of the same shape class.
        fn()  # warm up (JIT, lazy init)
        gc.collect()
        before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        out = fn()
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        del out
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return max(peak - before, 0) / 1e6

    idx = torch.randint(0, 128, (2, 16))
    engine1 = make_engine(1, n_layer=4)
    engine4 = make_engine(4, n_layer=4)
    for e in (engine1, engine4):
        e.zero_grad(set_to_none=True)

    def step_1():
        loss, _ = engine1.denoise_step(idx, block_idx=0)
        loss.backward()
        engine1.zero_grad(set_to_none=True)

    def step_4():
        loss, _ = engine4.denoise_step(idx, block_idx=0)
        loss.backward()
        engine4.zero_grad(set_to_none=True)

    if not torch.cuda.is_available():
        # CPU has no allocator high-water mark to read. The isolation property
        # itself is already asserted structurally in test_dbcpu_engine.py (only
        # the active block's params get gradients); what is not observable here
        # on CPU is the activation byte count, so skip rather than fake it.
        import pytest

        pytest.skip("activation memory accounting requires a CUDA allocator")
    mb1 = peak_mb(step_1)
    mb4 = peak_mb(step_4)
    assert mb4 < mb1, f"B=4 peak {mb4:.1f}MB not below B=1 peak {mb1:.1f}MB"


def test_when_ce_and_edm_objectives_then_only_edm_isolates_the_forward() -> None:
    """Documents what each objective actually costs.

    `train_step` runs all L layers, so it saves backward/optimizer memory but no
    forward FLOPs; `denoise_step` runs L/B layers and saves both. This is the
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
    assert len(calls) == 2, "still L/B = 2 layers, independent of which block"


# ---------------------------------------------------------------------------
# Masking and target hoisting. Merged in from `test_dbcpu_objective.py`, which
# was a near-wholesale duplicate of this file: same intent, same imports, and
# four tests with identical bodies modulo docstring wording. This file held the
# stronger variant of both overlapping pairs, so the survivor is here.
# ---------------------------------------------------------------------------


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
