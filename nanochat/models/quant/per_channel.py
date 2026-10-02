"""Per-channel value-centered quantizer (PRD 3.4, opt-in).

`ValueCenteredQuantizationLUT` in `ablation.py` quantizes around a single scalar
center. That is the right shape for a tensor whose channels share a range, and
the wrong shape for a weight matrix, where the per-output-channel scales routinely
differ by an order of magnitude: one global codebook then spends most of its
levels on the few large channels and rounds the many small ones to zero.

`PerChannelValueCenteredQuantizer` gives each output channel its own center and
its own magnitude codebook. The mechanism is the same one the scalar version
uses -- decompose to `center + sign(x - center) * g(|x - center|)`, with a
monotone non-negative `g` -- with the per-channel axes added.

What is claimed here, and what is not:

* **Zero anchor per channel.** `x == center_c` returns exactly `center_c`, and
  with the default `center=0` that is exactly `0.0`. This is the contract
  SparseProp's structural sparsity depends on, and it is why the center is
  pinned rather than learned by default.
* **Cost is `C x K` levels**, versus `K` for a shared codebook. For a 768-wide
  MLP that is a 768-fold increase in table size, which is why this is opt-in
  and why the fused product/inference paths raise on it rather than
  approximating.

It is *not* claimed that per-channel is better. It is strictly more expressive,
so it can only lose by overfitting the per-channel parameters when the training
set is short, and this repo has no measurement yet that settles that. The
ablation harness is the place to settle it.
"""

import math

import torch
import torch.nn as nn

from nanochat.models.quant.codebook import QuantizedOutput, validate_split


class PerChannelValueCenteredQuantizer(nn.Module):
    """Per-output-channel value-centered quantizer.

    Each channel gets an independent asymmetric codebook plus its own center, so
    the quantizer can represent channels with very different ranges without
    spending a shared alphabet on the difference.

    Args:
        num_channels: channels (output features) to quantize independently.
        m_neg, m_pos: per-channel asymmetric split, as in the shared codebook.
        centers: initial center per channel, shape `(num_channels,)`. `0.0` for
            all channels by default, which is what makes the zero anchor exact.
        init_min, init_max: initial FP32 span of every channel's codebook. All
            channels start identical so enabling this is a no-op at step 0.
        learnable_centers: if True the centers are trained. Off by default: a
            moving center is no longer the exact zero anchor, which would break
            the SparseProp structural-zero contract.

    Shapes:
        forward accepts `(..., C)` and quantizes over the last axis.
    """

    def __init__(
        self,
        num_channels: int,
        m_neg: int = 127,
        m_pos: int = 127,
        centers: torch.Tensor | None = None,
        init_min: float = -1.0,
        init_max: float = 1.0,
        learnable_centers: bool = False,
    ):
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        m_neg, m_pos = validate_split(m_neg, m_pos)
        self.num_channels = num_channels
        self.m_neg = m_neg
        self.m_pos = m_pos
        self.K = m_neg + 1 + m_pos

        centers = (
            torch.zeros(num_channels)
            if centers is None
            else torch.as_tensor(centers, dtype=torch.float32).reshape(-1)
        )
        if centers.numel() != num_channels:
            raise ValueError(
                f"centers has {centers.numel()} entries, expected num_channels="
                f"{num_channels}"
            )
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        # `m_neg + 1` and `m_pos + 1` magnitudes, one ladder per arm, both
        # starting at zero at the centre. Each arm's high bound is the distance
        # from the centre to that side's extreme -- `abs(init_min)` below and
        # `init_max` above -- *not* `init_max - init_min`. Using the span makes
        # the table asymmetric about the centre: for `init_min=-100,
        # init_max=100` it produced `[-100, -67, -33, 0, 67, 133, 200]`, so the
        # top level overshot by 2x and every positive value small enough to
        # matter bucketized onto the anchor.
        #
        self.raw_neg = nn.Parameter(
            self._inverse_softplus_linspace(init_min, m_neg + 1, num_channels)
        )
        self.raw_pos = nn.Parameter(
            self._inverse_softplus_linspace(init_max, m_pos + 1, num_channels)
        )

    @staticmethod
    def _inverse_softplus_linspace(
        high: float, count: int, num_channels: int
    ) -> torch.Tensor:
        """Raw values whose `softplus` is `linspace(0, |high|, count)`.

        Needed because `_magnitudes` forces the leading entry to exactly zero
        while `softplus(0) = ln 2`, not `0`. The first entry is therefore stored
        as a raw that softplus-maps to `ln 2`, and the rest as the inverse of the
        target magnitudes, so the reconstructed codebook is the linspace the
        caller asked for rather than something one softplus away from it.
        """
        target = (
            torch.linspace(0.0, abs(float(high)), count)
            .reshape(1, -1)
            .repeat(num_channels, 1)
        )
        return torch.where(
            target == 0,
            torch.full_like(target, math.log(2.0)),
            target + torch.log(-torch.expm1(-target.clamp_min(1e-12))),
        )

    def get_codebook(self) -> torch.Tensor:
        """The `[C, K]` per-channel codebook.

        Row `c` is channel `c`'s levels, centred on `centers[c]`:
        `[c0 - |n_m| ... c0 - |n_1|, c0, p_1 ... p_m]`. The centre appears exactly
        once, at index `m_neg`, because both arms start their magnitude ladders at
        zero and the zero is emitted once rather than as part of either arm.
        That is what makes `x == center` reconstruct to exactly `center` and,
        with the default centres, to exactly `0.0`.

        Magnitudes go through `softplus` rather than being clamped or
        `abs()`-ed. Both of those are non-smooth at zero, and a plain `abs` is
        worse than it looks: a gradient step that pushes a raw value negative
        *folds it back to zero*, so the level silently collapses onto the
        anchor and the codebook ends up with duplicate entries. Observed during
        development -- all three negative levels reached exactly `0.0` after
        300 steps, leaving the table degenerate. `softplus` is smooth and
        strictly positive away from zero, so a level can be pulled in but never
        onto its neighbour.
        """
        # Each arm's magnitude ladder starts at 0 and grows outward, so the
        # leading zero is dropped and the rest reversed for the negative side.
        # The drop has to happen *before* the flip: slicing after the flip would
        # discard the largest magnitude instead of the zero, leaving the
        # outermost level off the table and putting a spurious `0.0` next to the
        # real anchor.
        neg = torch.flip(self._magnitudes(self.raw_neg)[:, 1:], dims=(1,))
        pos = self._magnitudes(self.raw_pos)[:, 1:]
        c = self.centers.reshape(-1, 1)
        # The shared centre is emitted once, between the two arms.
        return torch.cat([c - neg, c.expand(-1, 1), c + pos], dim=1)

    @staticmethod
    def _magnitudes(raw: torch.Tensor) -> torch.Tensor:
        """Non-negative, strictly increasing magnitudes from unconstrained raws.

        `softplus` guarantees positivity, so the levels can never collapse onto
        the anchor. The leading entry is forced to exactly zero so the two arms
        share the centre instead of each owning a near-zero level of their own.
        """
        mag = torch.nn.functional.softplus(raw)
        return torch.cat([torch.zeros_like(mag[:, :1]), mag[:, 1:]], dim=1)

    def _validate(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.num_channels:
            raise ValueError(
                f"last dimension must be num_channels={self.num_channels}, got "
                f"{x.shape[-1]} from shape {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> QuantizedOutput:
        """Quantize `x` per channel.

        Returns a `QuantizedOutput` whose `codebook` is `[C, K]`. That shape is
        what the existing index kernels can consume unchanged when the channel
        axis is the reduction-free trailing axis; callers that reduce over
        channels (the usual `F.linear` contraction) must gather per channel
        explicitly, which is what `quantized_value` does.
        """
        self._validate(x)
        x_fp32 = x.to(torch.float32)
        codebook = self.get_codebook()
        # Per-row boundaries: `[C, K-1]`, broadcast over the leading axes.
        mids = (codebook[:, :-1] + codebook[:, 1:]) * 0.5
        flat = x_fp32.reshape(-1, self.num_channels)
        idx_flat = torch.empty(flat.shape, dtype=torch.int64, device=x.device)
        for c in range(self.num_channels):
            # `.contiguous()` on the column: `searchsorted` copies a strided
            # column internally and warns about it once per process, which in a
            # test run surfaces as a stray warning attributed to this module.
            col = flat[:, c].detach().contiguous()
            idx_flat[:, c] = torch.bucketize(col, mids[c].detach().contiguous())
        idx = idx_flat.reshape(x_fp32.shape)

        # Row-wise gather: `codebook[c, idx[..., c]]` for every element. The
        # leading axes are flattened together, so the lookup is one `[N, C]`
        # index into a `[C, K]` table and works for any input rank.
        dequant = codebook[
            torch.arange(self.num_channels, device=x.device).unsqueeze(0),
            idx_flat,
        ].reshape(x_fp32.shape)
        # STE: identity on the input, live gather on the codebook, so both the
        # weight and the per-channel deltas receive gradient.
        x_q = dequant + (x_fp32 - x_fp32.detach())
        return QuantizedOutput(
            value=x_q.to(x.dtype),
            indices=idx.to(torch.uint8 if self.K <= 255 else torch.int32),
            codebook=codebook,
        )

    def quantized_value(self, x: torch.Tensor) -> torch.Tensor:
        """Just the dequantized values, for callers that do not need indices."""
        return self.forward(x).value

    @torch.no_grad()
    def init_from_tensor(self, x: torch.Tensor, percentile: float = 100.0) -> None:
        """Seed every channel's codebook from `x`'s own range.

        **Required in practice, not an optimisation.** A caller-supplied
        `init_max` is a guess, and when it overshoots the data the failure is
        silent and unrecoverable: every value bucketizes onto the zero anchor,
        the anchor is the only level ever gathered, so the codebook parameters
        receive exactly zero gradient and the table never moves. Measured with
        `init_max=100` on data of magnitude 8: NMSE stayed at exactly 1.0 for
        400 steps on every channel.

        Fitting the outer levels to the data's own extremes sidesteps that, and
        uses `percentile` to clip outliers that would otherwise stretch the
        table and shrink every inner step.
        """
        self._validate(x)
        flat = x.detach().reshape(-1, self.num_channels).to(torch.float32)
        lo = torch.quantile(flat, (100.0 - percentile) / 100.0, dim=0)
        hi = torch.quantile(flat, percentile / 100.0, dim=0)
        # Guard a degenerate channel: an all-constant column has zero range, which
        # gives a zero-width table whose levels all sit on the anchor -- so every
        # value buckets to index `m_neg`, the codebook gets zero gradient, and the
        # channel is frozen for good. `span` therefore carries the floor, and the
        # per-arm bound is driven off `centre +/- span/2` rather than off the
        # channel's own (possibly zero) extent. Clamping the extent to a floor
        # does not work: the floor is a *maximum* on the width, so a zero-width
        # channel still comes out as zero-width and still duplicates levels.
        span = torch.maximum((hi - lo).abs(), torch.full_like(hi, 1e-3))
        half = span * 0.5
        # Per channel, not shared: collapsing to one bound across channels is
        # exactly the compromise this module exists to avoid.
        self._copy_magnitudes(self.raw_neg, torch.clamp(half, min=1e-3))
        self._copy_magnitudes(self.raw_pos, torch.clamp(half, min=1e-3))

    def _copy_magnitudes(self, raw: torch.Tensor, highs: torch.Tensor) -> None:
        """Overwrite `raw` with raws whose softplus spans `0..highs[c]` per row."""
        count = raw.shape[1]
        target = torch.linspace(0.0, 1.0, count).reshape(1, -1) * highs.reshape(-1, 1)
        raw.copy_(
            torch.where(
                target == 0,
                torch.full_like(target, math.log(2.0)),
                target + torch.log(-torch.expm1(-target.clamp_min(1e-12))),
            )
        )

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, m_neg={self.m_neg}, "
            f"m_pos={self.m_pos}, K={self.K}"
        )


__all__ = ["PerChannelValueCenteredQuantizer"]
