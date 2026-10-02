"""Tests for the per-channel value-centered quantizer (PRD 3.4).

Two things are load-bearing and get most of the attention here:

* **The exact centre anchor.** With the default zero centres, an input of `0.0`
  must reconstruct to exactly `0.0`, per channel. SparseProp prunes weights to
  that anchor and the CPU kernels skip on `w == 0.0`, so a quantizer that
  returns `1e-8` there would silently un-zero every pruned weight. An earlier
  draft of the centre was assembled with arithmetic that made this true only to
  floating-point precision, and the test below is what caught it.
* **Monotonicity per channel.** The levels must stay strictly increasing, or
  the midpoints stop being bucket boundaries and the gather stops being a gather.
"""

import pytest
import torch

from nanochat.models.quant.per_channel import PerChannelValueCenteredQuantizer


def quantizer(
    num_channels: int = 4, m_neg: int = 3, m_pos: int = 3
) -> PerChannelValueCenteredQuantizer:
    return PerChannelValueCenteredQuantizer(
        num_channels=num_channels,
        m_neg=m_neg,
        m_pos=m_pos,
        init_min=-2.0,
        init_max=2.0,
    )


class TestCodebookStructure:
    def test_when_built_then_the_codebook_is_one_row_per_channel(self) -> None:
        q = quantizer(num_channels=5)
        book = q.get_codebook()
        assert book.shape == (5, q.K)
        assert q.K == 7

    def test_when_built_then_every_channel_starts_identical(self) -> None:
        """Init equivalence: enabling this must not perturb step 0.

        All rows are the same by construction (the deltas are `repeat`ed), so
        switching the module on cannot confound a later comparison with an
        initialization change.
        """
        book = quantizer(num_channels=4).get_codebook()
        for c in range(1, 4):
            assert torch.equal(book[0], book[c])

    def test_when_built_then_the_centre_entry_is_exactly_the_centre(self) -> None:
        """The anchor must be exact, not merely close.

        This is the SparseProp contract, so it is checked as `== 0.0` rather than
        `allclose(..., atol=1e-6)`.
        """
        q = quantizer()
        book = q.get_codebook()
        assert torch.equal(book[:, q.m_neg], torch.zeros(4))

    def test_when_centres_are_set_then_the_anchor_is_the_centre(self) -> None:
        q = PerChannelValueCenteredQuantizer(
            num_channels=3, m_neg=3, m_pos=3, centers=torch.tensor([1.0, -2.0, 0.5])
        )
        book = q.get_codebook()
        assert torch.equal(book[:, q.m_neg], torch.tensor([1.0, -2.0, 0.5]))

    def test_when_built_then_the_levels_stay_strictly_increasing(self) -> None:
        q = quantizer()
        for c in range(q.num_channels):
            book = q.get_codebook()[c]
            assert bool(torch.all(book[1:] > book[:-1])), f"channel {c} not monotone"

    def test_when_built_then_the_anchor_is_at_index_m_neg(self) -> None:
        """The anchor's *position* is part of the contract too.

        `dispatch_index_linear` and the sparse kernels assume the zero level is
        index `m_neg`; an anchor stored elsewhere would be found as the wrong
        level entirely. Checked by position, not by value: index 0 is
        legitimately the most negative level, so the anchor is the *last*
        non-positive entry, not the minimum.
        """
        q = quantizer()
        book = q.get_codebook()
        for c in range(q.num_channels):
            assert torch.equal(book[c, q.m_neg], torch.zeros(()))
            # Nothing above the anchor is non-positive, so the exact-zero level
            # is unambiguously the last one at or below zero.
            assert bool(torch.all(book[c, q.m_neg + 1 :] > 0.0))
            assert bool(torch.all(book[c, : q.m_neg] < 0.0))


class TestQuantization:
    def test_when_quantizing_a_zero_then_it_returns_exactly_zero(self) -> None:
        q = quantizer()
        out = q(torch.zeros(2, 6, q.num_channels))
        assert torch.equal(out.value, torch.zeros(2, 6, q.num_channels))
        assert torch.equal(
            out.indices,
            torch.full((2, 6, q.num_channels), q.m_neg, dtype=out.indices.dtype),
        )

    def test_when_quantizing_then_the_indices_are_in_range(self) -> None:
        q = quantizer()
        out = q(torch.randn(3, 4, q.num_channels) * 10)
        assert int(out.indices.min()) >= 0
        assert int(out.indices.max()) < q.K

    def test_when_quantizing_then_a_large_value_lands_on_the_top_level(self) -> None:
        q = quantizer()
        out = q(torch.full((1, 2, q.num_channels), 1e6))
        assert torch.equal(
            out.indices,
            torch.full((1, 2, q.num_channels), q.K - 1, dtype=out.indices.dtype),
        )

    def test_when_quantizing_then_a_negative_value_lands_on_the_bottom_level(
        self,
    ) -> None:
        q = quantizer()
        out = q(torch.full((1, 2, q.num_channels), -1e6))
        assert torch.equal(
            out.indices, torch.zeros((1, 2, q.num_channels), dtype=out.indices.dtype)
        )

    def test_when_the_last_dim_is_wrong_then_it_raises(self) -> None:
        """A silent broadcast here would quantize a tensor that does not match
        the channel table, producing a plausible number that means nothing."""
        q = quantizer(num_channels=4)
        with pytest.raises(ValueError, match="num_channels"):
            q(torch.randn(2, 5))

    def test_when_the_num_channels_is_zero_then_it_raises(self) -> None:
        with pytest.raises(ValueError, match="num_channels"):
            PerChannelValueCenteredQuantizer(num_channels=0)

    def test_when_quantizing_2d_and_3d_then_shapes_are_preserved(self) -> None:
        q = quantizer()
        for shape in (
            (7, q.num_channels),
            (2, 3, q.num_channels),
            (2, 3, 4, q.num_channels),
        ):
            assert q(torch.randn(*shape)).value.shape == shape


class TestTraining:
    def test_when_training_then_gradients_reach_the_deltas(self) -> None:
        q = quantizer()
        q(torch.randn(4, q.num_channels) * 2.0).value.sum().backward()
        for name, p in q.named_parameters():
            assert p.grad is not None, f"{name} received no gradient"
            assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is zero"

    def test_when_training_then_gradients_reach_the_input(self) -> None:
        """The STE identity must survive; a detached forward kills this."""
        x = torch.randn(4, quantizer().num_channels, requires_grad=True)
        q = quantizer()
        q(x).value.sum().backward()
        assert x.grad is not None
        assert float(x.grad.abs().sum()) > 0.0

    def test_when_centres_are_learnable_then_they_train(self) -> None:
        q = PerChannelValueCenteredQuantizer(
            num_channels=3, m_neg=3, m_pos=3, learnable_centers=True
        )
        assert any(n == "centers" for n, _ in q.named_parameters())
        q(torch.randn(4, 3) * 2.0).value.sum().backward()
        assert q.centers.grad is not None
        assert float(q.centers.grad.abs().sum()) > 0.0

    def test_when_centres_are_fixed_then_they_are_not_parameters(self) -> None:
        """Off by default: a moving centre stops being the exact zero anchor,
        which is the SparseProp structural-zero contract."""
        q = quantizer()
        assert not any(n == "centers" for n, _ in q.named_parameters())

    def test_when_the_split_is_invalid_then_it_raises(self) -> None:
        with pytest.raises(ValueError):
            PerChannelValueCenteredQuantizer(num_channels=2, m_neg=-1, m_pos=4)


class TestInit:
    def test_when_a_channel_is_constant_then_init_still_leaves_it_trainable(
        self,
    ) -> None:
        """A constant column has zero range; the guard must not leave it collapsed.

        Without a width floor the table would be all-centre, every value
        bucketizes to the anchor, and -- like the wide-init case -- the channel
        gets zero gradient and never moves.
        """

    q = PerChannelValueCenteredQuantizer(
        num_channels=3, m_neg=3, m_pos=3, init_min=-1.0, init_max=1.0
    )
    x = torch.zeros(32, 3)
    x[:, 0] = torch.randn(32)  # only channel 0 has real range
    q.init_from_tensor(x)
    book = q.get_codebook()
    # Every channel stays strictly ordered, so no channel collapsed onto a point.
    for c in range(3):
        assert bool(torch.all(book[c, 1:] > book[c, :-1])), (
            f"channel {c} collapsed to a zero-width table: {book[c].tolist()}"
        )


class TestAgainstSharedCodebook:
    def test_when_a_channel_is_small_then_per_channel_reconstructs_it_better(
        self,
    ) -> None:
        """The reason to use this: a small channel beside a large one.

        A shared codebook has to span the largest channel, so the small ones are
        rounded onto the zero anchor -- NMSE 1.0, i.e. the channel is discarded
        entirely. A per-channel table is not forced to make that compromise.

        Measured **per channel**, not as a pooled mean. The pooled mean is
        dominated by the large channel, where the two are near-identical
        (0.1056 vs 0.1054), so a mean-based assertion is really only re-testing
        the big channel and would hide the entire effect. The measured result:
        channel 1 goes from 1.0 (discarded) to 0.398.
        """
        from nanochat.models.quant.codebook import MemoryEfficientLearnedCodebook

        # One channel 100x larger than the rest, which is the case a shared
        # alphabet handles worst.
        rows = []
        for c in range(4):
            scale = 100.0 if c == 0 else 1.0
            rows.append(torch.randn(256) * scale)
        x = torch.stack(rows, dim=1)  # [256, 4]
        x = x.relu().square()  # non-negative, as the MLP activation is

        per_channel = PerChannelValueCenteredQuantizer(
            num_channels=4, m_neg=3, m_pos=3, init_min=-100.0, init_max=100.0
        )
        shared = MemoryEfficientLearnedCodebook(
            m_neg=3, m_pos=3, init_min=-100.0, init_max=100.0
        )
        # Seed the per-channel table from the data. Without this the whole table
        # starts far wider than the data, every value lands on the anchor, and
        # the codebook gets exactly zero gradient -- the failure is silent and
        # permanent, not a slow start. The shared codebook needs no such help: a
        # scalar ladder is not per-channel, so a wide init only costs it
        # resolution, it does not zero its gradient.
        per_channel.init_from_tensor(x)

        # Both arms are *fitted to the same data for the same number of steps*,
        # under a per-channel normalized loss. Normalizing matters and is not a
        # thumb on the scale: with a pooled MSE the gradient is dominated by the
        # largest channel, and the small channels' tables simply never move --
        # measured, both arms then sit at NMSE 1.0 and the test measures nothing.
        # Per-channel normalization is also the only loss under which a
        # per-channel table is the right tool, so comparing under any other loss
        # would argue against the thing being compared.
        opt_pc = torch.optim.SGD(per_channel.parameters(), lr=1.0)
        opt_sh = torch.optim.SGD(shared.parameters(), lr=1.0)
        for _ in range(400):
            opt_pc.zero_grad()
            ((per_channel(x).value - x).square() / x.square().mean(0)).mean().backward()
            opt_pc.step()
            opt_sh.zero_grad()
            ((shared(x).value - x).square() / x.square().mean(0)).mean().backward()
            opt_sh.step()

        with torch.no_grad():
            pc_v, sh_v = per_channel(x).value, shared(x).value
            per_row = [
                float((pc_v[:, c] - x[:, c]).square().mean() / x[:, c].square().mean())
                for c in range(4)
            ]
            sh_row = [
                float((sh_v[:, c] - x[:, c]).square().mean() / x[:, c].square().mean())
                for c in range(4)
            ]

        # The large channel is where the two must agree -- the shared codebook is
        # free to spend its whole alphabet there, so per-channel cannot lose by
        # much and should not.
        assert per_row[0] <= sh_row[0] * 1.10 + 1e-6, (
            f"per-channel lost on the dominant channel: {per_row[0]:.4g} vs "
            f"{sh_row[0]:.4g}"
        )
        # The small channels are the point: at least one must be rescued from
        # being rounded entirely to the zero anchor.
        assert any(p < s - 0.1 for p, s in zip(per_row[1:], sh_row[1:])), (
            f"no small channel improved: per-channel {per_row[1:]} vs shared "
            f"{sh_row[1:]}"
        )
        # And the shared codebook really does discard them, so the test is
        # measuring a real effect rather than a marginal one.
        assert max(sh_row[1:]) > 0.9, (
            f"the shared codebook was expected to flatten the small channels, "
            f"got {sh_row[1:]}"
        )
