"""Slice 1: equi-probability noise partitioner + EDM preconditioning.

python -m pytest tests/test_dbcpu_partitioner.py -v
"""

import math

import pytest
import torch


def test_when_boundaries_then_endpoints_match_sigmas() -> None:
    from nanochat.diffusion_blocks import EquiProbabilityPartitioner

    p = EquiProbabilityPartitioner(num_blocks=4)
    b = p.boundaries()
    assert b.shape == (5,)
    assert b[0].item() == p.sigma_min
    assert b[-1].item() == p.sigma_max


def test_when_boundaries_then_strictly_increasing() -> None:
    from nanochat.diffusion_blocks import EquiProbabilityPartitioner

    b = EquiProbabilityPartitioner(num_blocks=4).boundaries()
    assert bool((b[1:] > b[:-1]).all())


def test_when_wide_range_then_middle_boundary_is_lognormal_median() -> None:
    from nanochat.diffusion_blocks import EquiProbabilityPartitioner

    p = EquiProbabilityPartitioner(
        num_blocks=2, sigma_min=1e-4, sigma_max=1e4, p_mean=-1.2, p_std=1.2
    )
    mid = p.boundaries()[1].item()
    assert mid == pytest.approx(math.exp(-1.2), abs=0.01)


def test_when_edm_factors_then_match_hand_computed_values() -> None:
    from nanochat.diffusion_blocks import edm_preconditioning

    cin, cout, w = edm_preconditioning(torch.tensor([0.5]), sigma_data=0.5)
    assert cin.item() == pytest.approx(1.4142135, rel=1e-4)
    assert cout.item() == pytest.approx(0.3535534, rel=1e-4)
    assert w.item() == pytest.approx(8.0, rel=1e-4)
