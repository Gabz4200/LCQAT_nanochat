"""
Tests for fused activation LUT compilation (PRD section 6).

python -m pytest tests/test_lcqat_lut.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.lcqat import compile_activation_lut


def make_codebook(K: int, span: float = 2.0) -> torch.Tensor:
    return torch.linspace(-span, span, K)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def test_when_compiling_then_lut_matches_direct_bucketization() -> None:
    in_cb = make_codebook(15)
    out_cb = make_codebook(15)
    lut = compile_activation_lut(relu_squared, in_cb, out_cb)
    midpoints = (out_cb[:-1] + out_cb[1:]) * 0.5
    expected = torch.bucketize(relu_squared(in_cb), midpoints)
    assert lut.dtype == torch.uint8
    assert lut.shape == (15,)
    assert torch.equal(lut.long(), expected)


def test_when_relu_squared_then_lut_is_monotonic_on_the_positive_branch() -> None:
    # relu^2 is nondecreasing on [0, inf), so target indices must not decrease
    # there; the negative branch maps everything to the anchor region.
    in_cb = make_codebook(15)
    out_cb = make_codebook(15)
    lut = compile_activation_lut(relu_squared, in_cb, out_cb).long()
    positive = in_cb >= 0
    pos_indices = lut[positive]
    assert (pos_indices[1:] >= pos_indices[:-1]).all()


def test_when_large_codebook_then_lut_uses_int32() -> None:
    in_cb = make_codebook(15)
    out_cb = make_codebook(257)
    lut = compile_activation_lut(relu_squared, in_cb, out_cb)
    assert lut.dtype == torch.int32


def test_when_codebook_is_2d_then_raises() -> None:
    cb = make_codebook(15).unsqueeze(0)
    with pytest.raises(ValueError, match="1-D"):
        compile_activation_lut(relu_squared, cb, make_codebook(15))


def test_when_codebook_too_small_then_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compile_activation_lut(relu_squared, torch.zeros(1), make_codebook(15))
