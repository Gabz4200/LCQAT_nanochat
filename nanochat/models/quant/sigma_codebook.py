"""Sigma-conditioned codebooks for DiffusionBlocks (PRD 3.2).

A DiffusionBlocks engine partitions sigma into B disjoint equi-probability
ranges and trains one block per range. The activation distribution of a layer
depends strongly on sigma: at `sigma ~ 0.002` the residual stream is close to
the clean embedding, at `sigma ~ 80` it is nearly pure noise. A single
activation codebook has to span both, so most of its levels are wasted on
values that never occur.

Two conditioning schemes are provided, and they answer different questions:

`SigmaConditionedCodebook` -- one codebook per *anchor sigma*, selected by a
hard bucket. Each block gets its own codebook, so the codebook matches the
block's noise range exactly. The cost is a factor `B` in codebook parameters
and in inference LUT size, and inference needs to know which codebook to read.

`SigmaModulatedCodebook` -- a single codebook whose levels are *shifted* by a
learned function of log(sigma). One set of parameters serves all noise levels,
so there is no B-fold parameter or LUT blow-up, and it stays usable in the fused
inference path, where the shift can be applied to the gathered values.

The choice matters because of the exported artifact: the two have different
storage contracts, and a claim about one is not a claim about the other.
"""

import math

import torch
import torch.nn as nn

from nanochat.models.quant.codebook import (
    MemoryEfficientLearnedCodebook,
    QuantizedOutput,
    validate_split,
)
from nanochat.models.quant.packing import index_dtype_for_k

# Log-sigma buckets are laid out symmetrically about log(SIGMA_PIVOT), which is
# the sigma at which the EDM preconditioner makes the data and noise terms
# comparable (`sigma_data`). Bucketing around that point means the boundaries
# are meaningful in the model's own units rather than in raw sigma, where
# everything below 1 and above 10 would be crowded together.
SIGMA_PIVOT = 0.5


def log_sigma_anchor_index(sigma: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Pick the nearest anchor for each `sigma` in log space.

    Args:
        sigma: noise levels, any shape.
        anchors: the codebook anchor sigmas, strictly increasing.

    Returns:
        int64 index into `anchors` for every element of `sigma`, same shape.

    Nearest-anchor rather than interval assignment, so every anchor owns a
    non-empty log-space region and no codebook is unreachable. That is what
    makes the hard-conditioned variant's guarantee ("each block sees its own
    codebook") actually true at the boundaries.
    """
    if anchors.numel() == 0:
        raise ValueError("anchors must be non-empty")
    if not bool(torch.all(anchors[1:] > anchors[:-1])):
        raise ValueError("anchors must be strictly increasing")
    log_a = torch.log(anchors.to(torch.float64))
    log_s = torch.log(sigma.to(torch.float64).clamp_min(1e-12))
    # bucketize is right-closed, so comparing log(s) against the midpoints in log
    # space yields the nearest anchor.
    mids = (log_a[:-1] + log_a[1:]) * 0.5
    return torch.bucketize(log_s, mids).to(torch.int64)


def resolve_batch_sigma(sigma: torch.Tensor, batch: int) -> torch.Tensor:
    """One sigma per batch row, shaped `(B, 1, 1)`.

    A single global sigma is a legitimate input, not a mistake: the
    DiffusionBlocks adapter is conditioned on the *step's* noise level rather
    than the row's, and passes `(1,)` against a `(B, 1, D)` activation. It is
    broadcast to the batch, so the conditioned codebook correctly sees the same
    levels for every row.

    Anything else -- a count matching neither 1 nor the batch -- raises rather
    than guessing, because a silent misread would condition every row on an
    arbitrary neighbour's noise level.
    """
    n = sigma.numel()
    if n == batch:
        return sigma.reshape(-1, 1, 1)
    if n == 1:
        return sigma.reshape(1).expand(batch).reshape(-1, 1, 1)
    raise ValueError(
        f"sigma must supply one value per batch element or exactly one shared "
        f"value: got {n} for x with batch {batch}"
    )


class SigmaConditionedCodebook(nn.Module):
    """One codebook per sigma anchor, selected by a hard log-sigma bucket.

    Parameters scale with the number of anchors: each holds its own
    `m_neg + 1 + m_pos` levels. That is the point (a codebook tuned to one
    noise range cannot be simultaneously wrong for all of them) and also the
    cost, so `num_anchors` is kept equal to the block count rather than
    increased for resolution.

    Args:
        num_anchors: number of distinct codebooks. Must be >= 2.
        m_neg, m_pos: per-codebook asymmetric split.
        anchors: strictly increasing anchor sigmas, length `num_anchors`.
        init_min, init_max: FP32 init span for *every* anchor. All anchors
            start identical, so enabling this module is a no-op at step 0
            relative to a shared codebook of the same K.
    """

    def __init__(
        self,
        num_anchors: int = 2,
        m_neg: int = 127,
        m_pos: int = 127,
        anchors: torch.Tensor | None = None,
        init_min: float = -1.0,
        init_max: float = 1.0,
        device=None,
    ):
        super().__init__()
        if num_anchors < 2:
            raise ValueError(f"num_anchors must be >= 2, got {num_anchors}")
        m_neg, m_pos = validate_split(m_neg, m_pos)
        self.m_neg = m_neg
        self.m_pos = m_pos
        self.K = m_neg + 1 + m_pos
        self.num_anchors = num_anchors
        if anchors is None:
            anchors = torch.logspace(
                math.log10(SIGMA_PIVOT) - 1.0,
                math.log10(SIGMA_PIVOT) + 1.0,
                num_anchors,
                device=device,
            )
        anchors = torch.as_tensor(anchors, dtype=torch.float32, device=device).reshape(
            -1
        )
        if anchors.numel() != num_anchors:
            raise ValueError(
                f"anchors has {anchors.numel()} entries, expected num_anchors="
                f"{num_anchors}"
            )
        if not bool(torch.all(anchors[1:] > anchors[:-1])):
            raise ValueError("anchors must be strictly increasing")
        # Persistent so a run with custom `anchors=` resumes with the same
        # bucket boundaries instead of silently re-bucketing to defaults.
        self.register_buffer("anchors", anchors, persistent=True)

        self.codebooks = nn.ModuleList(
            [
                MemoryEfficientLearnedCodebook(
                    m_neg=m_neg,
                    m_pos=m_pos,
                    init_min=init_min,
                    init_max=init_max,
                    device=device,
                )
                for _ in range(num_anchors)
            ]
        )

    def anchor_index(self, sigma: torch.Tensor) -> torch.Tensor:
        return log_sigma_anchor_index(sigma, self.anchors)

    def get_codebooks(self) -> torch.Tensor:
        """All codebooks stacked, shape `[num_anchors, K]`."""
        return torch.stack([cb.get_codebook() for cb in self.codebooks], dim=0)

    #: Read by `LCQATLinear.forward` to decide whether sigma is required.
    needs_sigma = True

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scale: float = 1.0,
    ) -> QuantizedOutput:
        """Quantize `x` with the codebook of its own noise level.

        Args:
            x: the tensor to quantize, shape `(B, T, D)`.
            sigma: noise level, shape `(B, 1, 1)`, `(B,)`, or a scalar for
                a single shared level.
            scale: PRD 2.4 codebook-gradient scale. Applied to the gathered
                value only, so `d/dx` keeps the STE identity.

        Returns:
            `QuantizedOutput` whose `codebook` is the per-row gather, so the
            downstream index kernels still get a single `[K]` table to fetch
            from; the rows differ only in which entries they hold.

        Raises:
            ValueError: if `sigma` provides neither one value per batch
                element nor exactly one shared value. See
                `resolve_batch_sigma`.
        """
        sigma = resolve_batch_sigma(sigma, x.shape[0])
        idx = self.anchor_index(sigma.reshape(-1))  # [B]
        per_row = [
            self.codebooks[int(i)](x[b], scale=scale)
            for b, i in enumerate(idx.tolist())
        ]
        value = torch.stack([q.value for q in per_row], dim=0)
        indices = torch.stack([q.indices for q in per_row], dim=0)
        # The per-row codebook gather is returned as a [K, B] table: index
        # kernels index the *last* axis by an integer, so this is the layout
        # that makes `codebook[indices]` resolve per row. Training and the
        # `_forward_quantized` path both rely on it.
        codebook = torch.stack([q.codebook for q in per_row], dim=1)  # [K, B]
        return QuantizedOutput(value=value, indices=indices, codebook=codebook)

    def bucketize(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = resolve_batch_sigma(sigma, x.shape[0])
        idx = self.anchor_index(sigma.reshape(-1))
        out = torch.empty(x.shape[:-1], dtype=torch.int64, device=x.device)
        # x is (B, ...) with one sigma row per batch element.
        for b, i in enumerate(idx.tolist()):
            out[b] = self.codebooks[int(i)].bucketize(x[b])
        return out

    def compile_for_inference(self) -> None:
        for cb in self.codebooks:
            cb.compile_for_inference()


class SigmaModulatedCodebook(nn.Module):
    """A single codebook, log(sigma)-modulated by a learned positive gain.

    Unlike `SigmaConditionedCodebook` this keeps one level table, so the
    exported artifact is the same size as the unconditional one. What changes is
    that levels are read as `C[j] * gain(log sigma)`: high-noise inputs have
    their levels stretched up, low-noise inputs compressed down.

    The conditioning is a *gain* rather than a shift, and it is not an
    interchangeable design choice -- see `effective_codebook` for the two
    contracts (exact zero anchor, strict monotonicity) that a shift cannot
    satisfy together.

    Args:
        m_neg, m_pos: the static codebook's split.
        hidden: width of the gain MLP. Small on purpose: this runs once per
            forward on a scalar input, and a large MLP would dominate the
            parameter budget it is meant to save.
    """

    def __init__(
        self,
        m_neg: int = 127,
        m_pos: int = 127,
        hidden: int = 64,
        init_min: float = -1.0,
        init_max: float = 1.0,
        device=None,
    ):
        super().__init__()
        m_neg, m_pos = validate_split(m_neg, m_pos)
        self.m_neg = m_neg
        self.m_pos = m_pos
        self.K = m_neg + 1 + m_pos
        self.base = MemoryEfficientLearnedCodebook(
            m_neg=m_neg,
            m_pos=m_pos,
            init_min=init_min,
            init_max=init_max,
            device=device,
        )
        # A two-layer MLP on log(sigma). The output layer starts at exactly
        # zero, so the gain is `exp(0) == 1.0` at init and the module is a
        # bit-exact drop-in for the unconditional codebook it replaces.
        self.gain_net = nn.Sequential(
            nn.Linear(1, hidden, device=device),
            nn.SiLU(),
            nn.Linear(hidden, hidden, device=device),
            nn.SiLU(),
            nn.Linear(hidden, 1, device=device),
        )
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)

    def effective_codebook(self, sigma: torch.Tensor) -> torch.Tensor:
        """`C * gain(sigma)`, shape `[..., K]`.

        The conditioning is a positive *gain*, not an additive shift, and that
        choice is forced by two contracts that a shift cannot satisfy at once:

        * **The zero anchor must stay exactly 0.0.** SparseProp prunes weights to
          the exact zero anchor so they become structural zeros with no post-hoc
          mask multiply, and the CPU kernels skip a slot on `w == 0.0`. An
          additive shift dequantizes a pruned weight to `shift` -- measured at
          0.014 / 0.19 / 12.0 across three noise levels -- so every pruned
          weight would contribute a real term and the sparse artifact would be
          silently wrong.
        * **The levels must stay strictly increasing.** A uniform additive shift
          preserves that, but *pinning* the anchor afterwards does not: with the
          arms moved up by 3.0, the top negative level lands at 2.857 and then
          exceeds the pinned 0.0 anchor. The codebook stops being sorted, the
          midpoints stop being boundaries, and the gather stops being a gather.

        A gain sidesteps the conflict entirely: the anchor is `0 * g == 0.0` for
        any `g > 0`, and scaling a strictly increasing codebook by a positive
        scalar keeps it strictly increasing. Both contracts hold by
        construction, for every gain, with no special-casing.
        """
        return self.base.get_codebook() * self.gain_for(sigma).unsqueeze(-1)

    def gain_for(self, sigma: torch.Tensor) -> torch.Tensor:
        """Positive gain, shape `[sigma.numel()]`.

        Flattened deliberately. Callers pass sigma as `[B, 1, 1]` (the shape the
        engine uses) and want a `[B, K]` codebook out; letting the extra
        singleton axes through would produce `[B, 1, 1, K]` and every later
        dimension would be wrong. The caller reshapes, so the batch dimension is
        always recoverable.
        """
        log_s = torch.log(sigma.to(torch.float32).clamp_min(1e-12)).reshape(-1, 1)
        raw = self.gain_net(log_s).reshape(-1)
        return torch.exp(raw.clamp(-4.0, 4.0))

    #: Read by `LCQATLinear.forward` to decide whether sigma is required.
    needs_sigma = True

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scale: float = 1.0,
    ) -> QuantizedOutput:
        sigma = resolve_batch_sigma(sigma, x.shape[0])
        x_fp32 = x.to(torch.float32)
        # Bucketize and gather against the *same* effective codebook, so the
        # bucket boundaries and the fetched values cannot drift apart.
        effective = self.effective_codebook(sigma)
        # A single sigma shared by the whole batch collapses to a 1-D table, so
        # the two ranks have to be handled separately: `gather` needs the index
        # to have the same rank as the table, which is not true of a shared
        # table against a 3-D index.
        if effective.shape[0] == 1:
            table = effective[0]
            idx = self.bucketize_against(table, x_fp32)
            dequant = table[idx]
            codebook = table
        else:
            idx = self.bucketize_against(effective, x_fp32)
            # Row-batched gather: the lookup is `effective[b, idx[b, t, d]]` for
            # every element. Flattening the token/feature axes into the gather
            # axis is what makes this a single call -- `gather` requires every
            # non-index dimension to match exactly, and `T`/`D` are free.
            dequant = effective.gather(1, idx.reshape(effective.shape[0], -1)).reshape(
                x_fp32.shape
            )
            codebook = effective
        # STE: identity on the input, live gather on the codebook. The gain's
        # own gradient arrives through `dequant` directly, which is what lets
        # the modulation learn.
        # `scale` shrinks only the codebook gradient (PRD 2.4 1/sqrt(N)); the
        # STE identity term is added after it, so the gradient reaching the
        # input activation is untouched.
        scaled = dequant * scale + (dequant * (1.0 - scale)).detach()
        x_q = scaled + (x_fp32 - x_fp32.detach())
        idx_dtype = index_dtype_for_k(self.K)
        return QuantizedOutput(
            value=x_q.to(x.dtype), indices=idx.to(idx_dtype), codebook=codebook
        )

    @staticmethod
    def bucketize_against(effective: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Bucketize `x` against a `[..., K]` codebook, broadcasting the batch.

        `effective` is `[K]` for a single sigma or `[B, K]` for one sigma per
        row; the result matches `x`'s shape either way.
        """
        if effective.dim() == 1:
            return torch.bucketize(x, (effective[:-1] + effective[1:]) * 0.5)
        # [B, K-1] boundaries broadcast over the token/feature axes.
        mids = (effective[:, :-1] + effective[:, 1:]) * 0.5
        # Flatten everything after the batch, bucketize each row, then restore
        # `x`'s exact shape. Rebuilding the shape from `x.shape[1:-1]` plus the
        # flattened width would double-count the last axis.
        flat = x.reshape(x.shape[0], -1)
        out = torch.empty(flat.shape, dtype=torch.int64, device=x.device)
        for b in range(flat.shape[0]):
            out[b] = torch.bucketize(flat[b], mids[b].detach())
        return out.reshape(x.shape)

    def bucketize(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        effective = self.effective_codebook(sigma)
        if effective.shape[0] == 1:
            effective = effective[0]
        return self.bucketize_against(effective, x.to(torch.float32))

    def compile_for_inference(self) -> None:
        """Freeze the static codebook; the shift MLP stays as a runtime cost.

        A compiled `SigmaModulatedCodebook` is no longer a drop-in for the
        static one, because the levels are no longer constant. Export has to
        keep the shift network alongside the frozen table.
        """
        self.base.compile_for_inference()


__all__ = [
    "SIGMA_PIVOT",
    "SigmaConditionedCodebook",
    "SigmaModulatedCodebook",
    "log_sigma_anchor_index",
]
