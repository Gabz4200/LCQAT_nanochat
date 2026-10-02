"""
Tests for the LC-QAT learned codebook primitive (PRD sections 2 and 3.1).

The codebook is asymmetric: `K = m_neg + 1 + m_pos` with the zero anchor at
index `m_neg` (PRD 2.1). `m_neg = 0` gives a one-sided codebook, which is the
right shape for `gpt.py`'s `relu(x).square()` MLP hidden tensor, and is what
makes even K (K=4, 8, 16, ...) legal.

python -m pytest tests/test_lcqat_codebook.py -v
"""

import pytest
import torch

from nanochat.models.quant import MemoryEfficientLearnedCodebook

# (m_neg, m_pos) shapes exercised below. The one-sided cases are the ones that
# motivated the asymmetry: m_neg=0 and m_pos=0 both give even K=8.
SPLITS = ((0, 7), (7, 0), (1, 1), (6, 8), (8, 6), (1, 0 + 1), (0, 2), (127, 127))


def make_codebook(
    m_neg: int = 7, m_pos: int = 7, init_min: float = -2.0, init_max: float = 2.0
) -> MemoryEfficientLearnedCodebook:
    torch.manual_seed(0)
    return MemoryEfficientLearnedCodebook(
        m_neg=m_neg, m_pos=m_pos, init_min=init_min, init_max=init_max
    )


def make_symmetric(K: int = 15, **kwargs) -> MemoryEfficientLearnedCodebook:
    torch.manual_seed(0)
    return MemoryEfficientLearnedCodebook.from_k(K, **kwargs)


def test_when_k_is_too_small_then_raises() -> None:
    with pytest.raises(ValueError, match=">= 3"):
        MemoryEfficientLearnedCodebook.from_k(1)


def test_when_split_is_negative_then_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MemoryEfficientLearnedCodebook(m_neg=-1, m_pos=4)


def test_when_one_sided_with_single_level_then_raises() -> None:
    # m_neg=0, m_pos=1 would put the zero anchor at the maximum level, leaving
    # no interior midpoint for bucketize on the positive side.
    with pytest.raises(ValueError, match="m_pos >= 2"):
        MemoryEfficientLearnedCodebook(m_neg=0, m_pos=1)
    with pytest.raises(ValueError, match="m_neg >= 2"):
        MemoryEfficientLearnedCodebook(m_neg=1, m_pos=0)


@pytest.mark.parametrize("m_neg,m_pos", SPLITS)
def test_when_constructing_then_codebook_is_monotonic_with_exact_zero_anchor(
    m_neg: int, m_pos: int
) -> None:
    cb = make_codebook(m_neg, m_pos)
    book = cb.get_codebook()
    assert book.shape == (m_neg + 1 + m_pos,)
    assert book.dtype == torch.float32
    assert torch.isfinite(book).all()
    # The anchor is at m_neg, not at K//2 -- this is the whole point of the
    # asymmetric split.
    assert book[m_neg].item() == 0.0
    assert (book[1:] > book[:-1]).all()


def test_when_one_sided_then_every_level_is_on_the_expected_side() -> None:
    nonneg = make_codebook(m_neg=0, m_pos=7).get_codebook()
    assert nonneg[0].item() == 0.0
    assert (nonneg >= 0).all()
    nonpos = make_codebook(m_neg=7, m_pos=0).get_codebook()
    assert nonpos[-1].item() == 0.0
    assert (nonpos <= 0).all()


@pytest.mark.parametrize("m_neg", [0, 1, 5, 7])
def test_when_exact_zero_input_then_index_is_the_anchor(m_neg: int) -> None:
    """The zero-anchor guarantee the sparsity path depends on.

    `bucketize(0.0, midpoints)` must return exactly `m_neg`: 0.0 is a *level*,
    so it lies strictly between two midpoints. If this ever regresses, pruning
    by zeroing a shadow weight stops producing structural zeros.
    """
    cb = make_codebook(m_neg, 7)
    assert cb.bucketize(torch.zeros(1)).item() == m_neg
    out = cb(torch.zeros(16))
    assert (out.indices == m_neg).all()
    assert (out.value == 0.0).all()


def test_when_zero_init_range_then_codebook_stays_finite_and_monotonic() -> None:
    # A zero span (e.g. zero-initialized weights) is floored rather than making
    # the inverse softplus diverge to -inf.
    cb = MemoryEfficientLearnedCodebook(m_neg=1, m_pos=1, init_min=0.0, init_max=0.0)
    book = cb.get_codebook()
    assert torch.isfinite(book).all()
    assert book[1].item() == 0.0
    assert (book[1:] > book[:-1]).all()


def test_when_init_span_is_large_then_inverse_softplus_stays_finite() -> None:
    # `log(expm1(delta))` overflows to inf above ~88 in FP32; the stable form
    # `delta + log(-expm1(-delta))` does not, so this constructs instead of
    # raising the way the original implementation did.
    cb = MemoryEfficientLearnedCodebook(m_neg=1, m_pos=2, init_min=-1e4, init_max=1e4)
    book = cb.get_codebook()
    assert torch.isfinite(book).all()
    assert (book[1:] > book[:-1]).all()


def test_when_from_k_then_split_is_symmetric() -> None:
    cb = make_symmetric(15)
    assert (cb.m_neg, cb.m_pos) == (7, 7)
    assert cb.K == 15
    # An even K splits evenly rather than raising.
    even = make_symmetric(8)
    assert (even.m_neg, even.m_pos) == (3, 4)
    assert even.K == 8


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


def test_when_k_exceeds_255_then_indices_are_int32() -> None:
    out = make_symmetric(511)(torch.randn(8))
    assert out.indices.dtype == torch.int32


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


def test_when_one_sided_then_missing_branch_is_none_and_still_forwards() -> None:
    cb = make_codebook(m_neg=0, m_pos=7)
    assert cb.raw_neg_deltas is None
    x = (torch.rand(32) * 2.0).requires_grad_()
    out = cb(x)
    assert (out.value >= 0).all()
    out.value.sum().backward()
    assert cb.raw_pos_deltas.grad.abs().sum() > 0


def test_when_adversarial_optimizer_steps_then_ordering_survives() -> None:
    """No optimizer state, however violent, can invert the level order.

    The softplus prefix-sum parameterization is what makes this unbreakable by
    construction: there is no sort, no clamp, and no projection an update has to
    survive, because the ordering is a property of the function rather than of
    the parameter values.

    The precise claim is *monotone non-decreasing*, not *strictly increasing*:
    softplus is strictly increasing and strictly positive, so distinct latent
    deltas always give distinct levels, but two latent deltas that happen to be
    equal give equal increments and therefore equal levels. Equality is
    reachable (two randn draws can collide, and softplus saturates to the
    identity above |rho| ~ 30 in any precision), so asserting strictness would
    be asserting something false. What the parameterization does rule out --
    and what a sort/clamp scheme has to actively defend -- is inversion.
    """
    cb = make_codebook(7, 7)
    opt = torch.optim.AdamW([cb.raw_neg_deltas, cb.raw_pos_deltas], lr=1.0)
    for _ in range(100):
        opt.zero_grad()
        cb(torch.randn(64)).value.pow(2).mean().backward()
        opt.step()
        with torch.no_grad():
            cb.raw_neg_deltas.add_(torch.randn_like(cb.raw_neg_deltas) * 4.0)
            cb.raw_pos_deltas.add_(torch.randn_like(cb.raw_pos_deltas) * 4.0)
        book = cb.get_codebook()
        assert torch.isfinite(book).all()
        assert (book[1:] >= book[:-1]).all(), "levels inverted"
        assert book[cb.m_neg].item() == 0.0


def test_when_latent_deltas_are_distinct_then_levels_are_strictly_increasing() -> None:
    """Strict ordering does hold -- it just needs distinct increments.

    Isolation of the guarantee: with strictly increasing latent deltas in a
    regime where softplus resolves, every level is strictly ordered, no matter
    what the optimizer did to get there.
    """
    cb = make_codebook(7, 7)
    opt = torch.optim.AdamW([cb.raw_neg_deltas, cb.raw_pos_deltas], lr=1.0)
    for _ in range(50):
        opt.zero_grad()
        cb(torch.randn(64)).value.pow(2).mean().backward()
        opt.step()
        with torch.no_grad():
            # Monotone-but-distinct perturbation: every level keeps a strictly
            # positive, strictly unequal increment, however large the magnitudes.
            for p in (cb.raw_neg_deltas, cb.raw_pos_deltas):
                p.add_(torch.arange(p.numel(), dtype=p.dtype) / p.numel() * 3.0)
        book = cb.get_codebook()
        assert torch.isfinite(book).all()
        assert (book[1:] > book[:-1]).all(), "levels inverted or duplicated"
        assert book[cb.m_neg].item() == 0.0


def test_when_latent_deltas_saturate_then_levels_merge_but_never_invert() -> None:
    """Documents the precision regime, which is a real constraint.

    `softplus(rho)` saturates to the identity above |rho| ~ 30 in float32 *and*
    in float64, so once the latent deltas reach the tens, equal inputs give
    equal increments and adjacent levels merge. float32 additionally merges
    levels whose increment falls below one ULP of the running sum. In both
    regimes levels merge; they never invert, because the increments are still
    non-negative.

    The guarantee that survives unconditionally is the exact zero anchor: it is
    an index assignment rather than a computed value, which is precisely what
    the sparsity path relies on when it prunes by writing an exact 0.0.
    """
    cb = make_codebook(7, 7)
    with torch.no_grad():
        cb.raw_pos_deltas.fill_(60.0)
        cb.raw_neg_deltas.fill_(60.0)
    book = cb.get_codebook()
    assert book[cb.m_neg].item() == 0.0  # structural, always exact
    assert torch.isfinite(book).all()
    assert (book[1:] >= book[:-1]).all()  # merged, never inverted
    # The exact-zero contract the pruning path depends on still holds.
    assert cb.bucketize(torch.zeros(1)).item() == cb.m_neg


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
    # Attributes still resolve (to None) so split introspection keeps working,
    # but they are no longer parameters and leave the state_dict.
    assert cb.raw_pos_deltas is None
    assert cb.raw_neg_deltas is None
    assert "raw_pos_deltas" not in dict(cb.named_parameters())
    assert torch.equal(expected, cb(x).value)
    state = cb.state_dict()
    assert "raw_pos_deltas" not in state
    assert "compiled_codebook" in state


def test_when_compiled_one_sided_then_forward_still_matches() -> None:
    cb = make_codebook(m_neg=0, m_pos=7)
    x = torch.rand(32) * 2.0
    expected = cb(x).value
    cb.compile_for_inference()
    assert torch.equal(expected, cb(x).value)
    assert cb.raw_neg_deltas is None


def test_when_random_inputs_then_outputs_are_finite() -> None:
    cb = make_symmetric(255, init_min=-5.0, init_max=5.0)
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
