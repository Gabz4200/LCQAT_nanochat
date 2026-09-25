"""
Tests for the LC-QAT learned codebook primitive (PRD sections 2 and 3.1).

python -m pytest tests/test_lcqat_codebook.py -v
"""

import pytest
import torch

from nanochat.lcqat import MemoryEfficientLearnedCodebook


def make_codebook(
    K: int = 15, init_min: float = -2.0, init_max: float = 2.0
) -> MemoryEfficientLearnedCodebook:
    torch.manual_seed(0)
    return MemoryEfficientLearnedCodebook(K=K, init_min=init_min, init_max=init_max)


def test_when_k_is_even_then_raises() -> None:
    with pytest.raises(ValueError, match="odd"):
        MemoryEfficientLearnedCodebook(K=16)


def test_when_k_is_too_small_then_raises() -> None:
    with pytest.raises(ValueError, match="odd"):
        MemoryEfficientLearnedCodebook(K=1)


def test_when_constructing_then_codebook_is_monotonic_with_exact_zero_anchor() -> None:
    for K in (3, 15, 255):
        cb = make_codebook(K)
        book = cb.get_codebook()
        assert book.shape == (K,)
        assert book.dtype == torch.float32
        assert torch.isfinite(book).all()
        assert book[K // 2].item() == 0.0
        assert (book[1:] > book[:-1]).all()


def test_when_zero_init_range_then_codebook_stays_finite_and_monotonic() -> None:
    cb = MemoryEfficientLearnedCodebook(K=3, init_min=0.0, init_max=0.0)
    book = cb.get_codebook()
    assert torch.isfinite(book).all()
    assert book[1].item() == 0.0
    assert (book[1:] > book[:-1]).all()


def test_when_forward_then_value_is_exact_codebook_lookup() -> None:
    cb = make_codebook()
    book = cb.get_codebook()
    x = torch.randn(128)
    out = cb(x)
    assert out.value.shape == x.shape
    assert out.value.dtype == x.dtype
    assert out.indices.dtype == torch.uint8
    assert int(out.indices.max()) < cb.K
    assert torch.equal(out.value, book[out.indices.long()])


def test_when_backward_then_input_gradient_is_identity() -> None:
    cb = make_codebook()
    x = torch.randn(64, requires_grad=True)
    cb(x).value.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_when_backward_on_positive_inputs_then_only_positive_branch_receives_grad() -> (
    None
):
    cb = make_codebook()
    x = (torch.rand(64) * 3.0 + 0.5).requires_grad_()
    cb(x).value.sum().backward()
    assert cb.raw_pos_deltas.grad is not None
    assert cb.raw_pos_deltas.grad.abs().sum() > 0
    assert cb.raw_neg_deltas.grad is not None
    assert cb.raw_neg_deltas.grad.abs().sum() == 0


def test_when_backward_on_negative_inputs_then_only_negative_branch_receives_grad() -> (
    None
):
    cb = make_codebook()
    x = (-(torch.rand(64) * 3.0 + 0.5)).requires_grad_()
    cb(x).value.sum().backward()
    assert cb.raw_neg_deltas.grad is not None
    assert cb.raw_neg_deltas.grad.abs().sum() > 0
    assert cb.raw_pos_deltas.grad is not None
    assert cb.raw_pos_deltas.grad.abs().sum() == 0


def test_when_zero_input_then_value_is_exact_anchor() -> None:
    cb = make_codebook()
    out = cb(torch.zeros(16))
    assert (out.indices == cb.m).all()
    assert (out.value == 0).all()


def test_when_eval_mode_then_codebook_still_tracks_parameters() -> None:
    # Regression against PRD 3.1's eval-mode auto-cache: caching the compiled
    # codebook on the first eval forward would freeze codebook gradients for
    # trainers that run forward passes under model.eval() (scripts/chat_rl.py does).
    cb = make_codebook()
    cb.eval()
    x = torch.randn(32, requires_grad=True)
    cb(x).value.sum().backward()
    assert cb.raw_pos_deltas.grad is not None
    assert cb.raw_pos_deltas.grad.abs().sum() > 0
    book_before = cb.get_codebook().clone()
    with torch.no_grad():
        cb.raw_pos_deltas.add_(1.0)
    assert not torch.equal(book_before, cb.get_codebook())


def test_when_compiled_then_forward_matches_and_parameters_are_removed() -> None:
    cb = make_codebook()
    x = torch.randn(32)
    expected = cb(x).value
    cb.compile_for_inference()
    assert not hasattr(cb, "raw_pos_deltas")
    assert not hasattr(cb, "raw_neg_deltas")
    assert torch.equal(expected, cb(x).value)
    state = cb.state_dict()
    assert "raw_pos_deltas" not in state
    assert "compiled_codebook" in state


def test_when_random_inputs_then_outputs_are_finite() -> None:
    cb = make_codebook(K=255, init_min=-5.0, init_max=5.0)
    out = cb(torch.randn(1000))
    assert torch.isfinite(out.value).all()
    assert torch.isfinite(out.codebook).all()
    assert int(out.indices.max()) < 255


def test_when_bf16_input_then_dtype_is_preserved() -> None:
    cb = make_codebook()
    x = torch.randn(32, dtype=torch.bfloat16)
    out = cb(x)
    assert out.value.dtype == torch.bfloat16
    assert out.codebook.dtype == torch.float32
