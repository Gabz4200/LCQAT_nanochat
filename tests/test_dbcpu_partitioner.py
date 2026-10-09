"""Slice 1: equi-probability noise partitioner + EDM preconditioning.

python -m pytest tests/test_dbcpu_partitioner.py -v
"""

import math

import pytest
import torch


def test_when_boundaries_then_endpoints_match_sigmas() -> None:
    from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

    p = EquiProbabilityPartitioner(num_blocks=4)
    b = p.boundaries()
    assert b.shape == (5,)
    assert b[0].item() == p.sigma_min
    assert b[-1].item() == p.sigma_max


def test_when_boundaries_then_strictly_increasing() -> None:
    from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

    b = EquiProbabilityPartitioner(num_blocks=4).boundaries()
    assert bool((b[1:] > b[:-1]).all())


def test_when_wide_range_then_middle_boundary_is_lognormal_median() -> None:
    from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

    p = EquiProbabilityPartitioner(
        num_blocks=2, sigma_min=1e-4, sigma_max=1e4, p_mean=-1.2, p_std=1.2
    )
    mid = p.boundaries()[1].item()
    assert mid == pytest.approx(math.exp(-1.2), abs=0.01)


def test_when_edm_factors_then_match_hand_computed_values() -> None:
    from nanochat.training.diffusion_blocks import edm_preconditioning

    cin, cout, w = edm_preconditioning(torch.tensor([0.5]), sigma_data=0.5)
    assert cin.item() == pytest.approx(1.4142135, rel=1e-4)
    assert cout.item() == pytest.approx(0.3535534, rel=1e-4)
    assert w.item() == pytest.approx(8.0, rel=1e-4)


def test_when_sample_sigma_then_samples_stay_in_the_block_range() -> None:
    """``sample_sigma`` draws from the block's own interval, in CDF mass.

    A sampler that returned a constant inside the range would pass a
    finiteness check but silently collapse the block's noise levels to
    one value. The samples must stay inside ``range_for_block`` and
    actually vary across draws.
    """
    from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

    torch.manual_seed(0)
    p = EquiProbabilityPartitioner(num_blocks=4)
    lo, hi = (v.item() for v in p.range_for_block(1))
    samples = torch.stack(
        [
            p.sample_sigma(1, generator=torch.Generator().manual_seed(s))
            for s in range(64)
        ]
    )
    assert samples.shape == (64,)
    assert bool((samples >= lo).all()), "sample below block range"
    assert bool((samples <= hi).all()), "sample above block range"
    # The draws are not all the same value: a degenerate sampler that returned the range midpoint would pass the bounds check but collapse the noise distribution.
    assert float(samples.std()) > 0.0, "sample_sigma returned a constant"


def test_when_sample_sigma_then_higher_blocks_draw_higher_noise() -> None:
    """Block 0 owns the highest noise range (paper Fig. 6 / App. C).

    ``range_for_block`` is the single owner of the mapping, so the
    samples drawn for block 0 must dominate those for the last block.
    A reversed or scrambled mapping would train the wrong noise level
    onto the wrong layers while every single-sample test still passes.
    """
    from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

    torch.manual_seed(1)
    p = EquiProbabilityPartitioner(num_blocks=4)
    first = torch.stack(
        [
            p.sample_sigma(0, generator=torch.Generator().manual_seed(s))
            for s in range(32)
        ]
    )
    last = torch.stack(
        [
            p.sample_sigma(3, generator=torch.Generator().manual_seed(s))
            for s in range(32)
        ]
    )
    assert float(first.mean()) > float(last.mean()), (
        "block 0 must draw higher noise than the last block"
    )
