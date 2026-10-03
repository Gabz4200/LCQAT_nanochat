"""
Asymmetric learned codebook quantization (LC-QAT PRD sections 2 and 3.1).

Pure tensor math: no training loops, no hardware logic, no config objects.
The parameter names `raw_pos_deltas` / `raw_neg_deltas` are a stable contract:
`nanochat.models.quant.optimizer.build_qat_param_groups` splits them into their own
optimizer group by name. (`GPT.setup_optimizer` used to, and was deleted: it only
ever saw `transformer.h`, so it would have dropped the DiffusionBlocks adapters
and per-block denoise heads.)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.quant.packing import index_dtype_for_k

#: Smallest legal level count. K = m_neg + 1 + m_pos with both sides >= 1 gives
#: K >= 3; a one-sided codebook (m_neg = 0) needs m_pos >= 2 so the zero anchor
#: is not also the maximum level (which would make every positive input clamp).
MIN_CODEBOOK_K = 3


def validate_split(m_neg: int, m_pos: int) -> tuple[int, int]:
    """Validate an (m_neg, m_pos) level split and return it as ints.

    Raises:
        ValueError: on non-integer, negative, or too-small inputs.
    """
    for name, value in (("m_neg", m_neg), ("m_pos", m_pos)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    # A zero anchor that is also an endpoint gives bucketize no interior
    # midpoint on that side, so a one-sided codebook needs two positive levels.
    if m_pos == 0 and m_neg < 2:
        raise ValueError(
            "m_pos=0 requires m_neg >= 2 so the zero anchor is not also the "
            f"minimum level; got m_neg={m_neg}, m_pos={m_pos}"
        )
    if m_neg == 0 and m_pos < 2:
        raise ValueError(
            "m_neg=0 requires m_pos >= 2 so the zero anchor is not also the "
            f"maximum level; got m_neg={m_neg}, m_pos={m_pos}"
        )
    if m_neg + 1 + m_pos < MIN_CODEBOOK_K:
        raise ValueError(
            f"codebook K = m_neg + 1 + m_pos must be >= {MIN_CODEBOOK_K}, got "
            f"{m_neg + 1 + m_pos} (m_neg={m_neg}, m_pos={m_pos})"
        )
    return m_neg, m_pos


def _inverse_softplus(delta: torch.Tensor) -> torch.Tensor:
    """Return `rho` such that `F.softplus(rho) == delta`, for `delta > 0`.

    Uses `delta + log(-expm1(-delta))` rather than `log(expm1(delta))` so spans
    above ~88 in FP32 (where `expm1` overflows to inf) stay finite.

    Callers whose target can be an exact zero need their own zero handling --
    see `per_channel.raw_for_magnitudes`, which maps a zero target onto the raw
    whose softplus is `ln 2`.
    """
    return delta + torch.log(-torch.expm1(-delta))


def split_from_k(K: int) -> tuple[int, int]:
    """The `(m_neg, m_pos)` an integer cardinality `K` splits into.

    An odd K splits symmetrically (`m_neg == m_pos`); an even K cannot, so the
    extra level goes to the positive side. This is the pre-asymmetric API's one
    rule, shared by `from_k` and by `retrofit.spec_split`, which has to answer
    the same question about a config value without building a codebook.
    """
    if isinstance(K, bool) or not isinstance(K, int):
        raise ValueError(f"K must be an int, got {K!r}")
    if K < MIN_CODEBOOK_K:
        raise ValueError(f"Codebook size K must be >= {MIN_CODEBOOK_K}, got {K}")
    m_neg = (K - 1) // 2
    return m_neg, K - 1 - m_neg


def codebook_midpoints(codebook: torch.Tensor) -> torch.Tensor:
    """Bucketize boundaries of a `[K]` level table: the `[K-1]` midpoints.

    One definition of the boundary set, because the zero-anchor guarantee is a
    property of *these* values: with `m_neg` levels below zero and the anchor
    at index `m_neg`, `bucketize(0.0, midpoints)` returns exactly `m_neg` for
    any K, and that is what makes a pruned (exactly 0.0) shadow weight a
    structural zero with no post-hoc mask multiply. A second copy of the
    expression is a second chance to spell it differently.
    """
    return (codebook[:-1] + codebook[1:]) * 0.5


@dataclass
class QuantizedOutput:
    """Typed output contract for a quantization step (PRD 2.2)."""

    value: torch.Tensor  # STE-quantized tensor in the input dtype
    indices: torch.Tensor  # codebook indices, uint8 (K <= 255) or int32
    codebook: torch.Tensor  # FP32 codebook of shape (K,) used for this quantization


class MemoryEfficientLearnedCodebook(nn.Module):
    """
    Parametric asymmetric learned codebook of cardinality K = m_neg + 1 + m_pos.

    Index m_neg is anchored strictly to FP32 0.0. Levels are cumulative softplus
    step sizes, so gradient updates cannot break the level ordering (PRD 1.1).
    Forward emits the dequantized value with a straight-through estimator that
    routes gradients to both the input and the codebook (PRD 1.3).

    Ordering guarantee, precisely: softplus is strictly increasing and strictly
    positive, so the prefix sum is *monotone non-decreasing* for arbitrary latent
    deltas and *strictly increasing* whenever the latent deltas are distinct. The
    guarantee is that no optimizer update can **invert** the order -- there is no
    sort, clamp, or projection involved. It is not that levels are always
    distinct in floating point: two equal latent deltas give equal increments,
    and softplus saturates to the identity above |rho| ~ 30 in any precision.
    Distinctness is a training-regime property, not a structural one.

    The exact zero anchor, by contrast, *is* structural: it is an index
    assignment rather than a computed value, so `bucketize(0.0)` returns exactly
    m_neg in any regime. That is what makes pruning by writing an exact 0.0 into
    a shadow weight produce a structural zero.

    The two sides are independent (PRD 2.1: K = M_neg + 1 + M_pos), so a
    non-negative tensor can be quantized with m_neg = 0 and spend every level on
    the half of the range it actually occupies. `gpt.py`'s MLP applies
    `relu(x).square()` before `mlp.c_proj`, so the 4*n_embd hidden tensor -- the
    largest activation in the model -- is non-negative, and a symmetric codebook
    would waste half its levels. The split is asymmetric for exactly that reason.

    Args:
        m_neg: number of negative levels. 0 gives a one-sided (non-negative)
            codebook, which is legal and exact.
        m_pos: number of positive levels.
        init_min: initial value of the lowest level (only used when m_neg > 0).
        init_max: initial value of the highest level.
        device: optional device for the step parameters.
    """

    def __init__(
        self,
        m_neg: int = 127,
        m_pos: int = 127,
        init_min: float = -1.0,
        init_max: float = 1.0,
        device=None,
    ):
        super().__init__()
        m_neg, m_pos = validate_split(m_neg, m_pos)
        self.m_neg = m_neg
        self.m_pos = m_pos
        self.K = m_neg + 1 + m_pos

        # Floor the init span: a zero span (e.g. zero-initialized weights) would
        # make inverse softplus diverge to -inf and freeze the codebook at 0.0.
        span_pos = max(abs(init_max), 1e-6) if m_pos > 0 else 0.0
        span_neg = max(abs(init_min), 1e-6) if m_neg > 0 else 0.0

        if m_pos > 0:
            init_pos = torch.linspace(0, span_pos, m_pos + 1, device=device)[1:]
            pos_deltas = init_pos - torch.cat([init_pos.new_zeros(1), init_pos[:-1]])
            # Inverse softplus, in the numerically stable form
            # `delta + log(-expm1(-delta))`: `log(expm1(delta))` overflows to inf
            # for spans above ~88 in FP32, which would make large-init codebooks
            # unconstructible. Same convention as nanochat.models.quant.ablation.
            self.raw_pos_deltas = nn.Parameter(_inverse_softplus(pos_deltas))
        else:
            self.register_parameter("raw_pos_deltas", None)

        if m_neg > 0:
            init_neg = torch.linspace(0, span_neg, m_neg + 1, device=device)[1:]
            neg_deltas = init_neg - torch.cat([init_neg.new_zeros(1), init_neg[:-1]])
            self.raw_neg_deltas = nn.Parameter(_inverse_softplus(neg_deltas))
        else:
            self.register_parameter("raw_neg_deltas", None)

        # Static FP32 LUT for deployment; only read after compile_for_inference().
        #
        # `zeros`, not `empty`: this buffer is persistent, so it is written into
        # every checkpoint. With `empty` the saved tensor is uninitialized memory,
        # which in practice means NaN garbage for every codebook in every saved
        # model -- harmless at runtime (the values are overwritten by
        # compile_for_inference and never read before then) but it makes
        # save/load round-trip comparisons fail unpredictably, and it ships
        # meaningless bytes.
        self.register_buffer(
            "compiled_codebook",
            torch.zeros(self.K, dtype=torch.float32, device=device),
            persistent=True,
        )
        self.is_compiled = False

    def get_codebook(self) -> torch.Tensor:
        # Deliberately no eval-mode auto-caching (PRD 3.1 caches on first eval
        # forward): that would freeze codebook gradients for any trainer that
        # runs forward passes under model.eval() (chat_rl does) and would mutate
        # buffers inside torch.compile graphs. Freezing happens only explicitly,
        # via compile_for_inference().
        if self.is_compiled:
            return self.compiled_codebook
        parts = []
        if self.m_neg > 0:
            neg_steps = F.softplus(self.raw_neg_deltas)
            parts.append((-torch.cumsum(neg_steps, dim=0)).flip(0))
        if self.m_pos > 0:
            pos_steps = F.softplus(self.raw_pos_deltas)
            parts.append(torch.cumsum(pos_steps, dim=0))
        # The zero anchor sits between the two sides, so it takes its device and
        # dtype from whichever side exists; at least one always does by
        # construction (`validate_split` requires m_neg + m_pos >= 2).
        anchor = parts[0].new_zeros(1) if parts else self.compiled_codebook.new_zeros(1)
        if self.m_pos > 0:
            # [neg..., 0.0, pos...]
            return torch.cat([*parts[:-1], anchor, parts[-1]])
        # m_pos == 0: only the negative side exists, and it is already ascending,
        # so the anchor goes on its right.
        return torch.cat([*parts, anchor])

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> QuantizedOutput:
        """Quantize `x`, optionally scaling only the codebook gradient.

        `scale` exists so the sigma-conditioned wrappers can thread the PRD 2.4
        `1/sqrt(N)` factor down to a codebook they do not own. It multiplies the
        gathered value *before* the STE identity term is added, so it shrinks
        `dL/dC` while leaving `dL/dx` at exactly 1.0 -- applying it afterwards
        would also attenuate the input gradient, which is a silent change to how
        the model trains.
        """
        codebook = self.get_codebook().to(torch.float32)
        x_fp32 = x.to(torch.float32)

        indices = torch.bucketize(x_fp32.detach(), codebook_midpoints(codebook))
        x_dequant = codebook[indices]
        if scale != 1.0:
            x_dequant = x_dequant * scale + (x_dequant * (1.0 - scale)).detach()

        # STE (PRD 1.3): forward value is exactly C[Q]; the zero-valued identity
        # term routes dL/dx, and the live gather routes dL/dC = scatter of dL/dy
        # over assigned buckets. A plain `x + (C[Q] - x).detach()` would zero out
        # the codebook gradient.
        x_q = x_dequant + (x_fp32 - x_fp32.detach())

        idx_dtype = index_dtype_for_k(self.K)
        return QuantizedOutput(
            value=x_q.to(x.dtype),
            indices=indices.to(idx_dtype),
            codebook=codebook,
        )

    def bucketize(self, x: torch.Tensor) -> torch.Tensor:
        """Return the codebook index of each element of `x` (no value gather).

        Exposed because the zero-anchor guarantee is load-bearing for the
        sparsity path: with `m_neg` levels below zero and the zero anchor at
        index `m_neg`, `bucketize(0.0, midpoints)` returns exactly `m_neg`,
        since 0.0 is a *level* and therefore lies strictly between two
        midpoints. That is what lets a pruned (exactly 0.0) shadow weight become
        a structural zero with no post-hoc mask multiply.
        """
        codebook = self.get_codebook().to(torch.float32)
        return torch.bucketize(
            x.detach().to(torch.float32), codebook_midpoints(codebook)
        )

    @classmethod
    def from_k(
        cls,
        K: int,
        init_min: float = -1.0,
        init_max: float = 1.0,
        device=None,
    ) -> "MemoryEfficientLearnedCodebook":
        """Build a codebook of total size `K` (the pre-asymmetric API).

        An odd K splits symmetrically (`m_neg == m_pos`); an even K cannot, so
        the extra level goes to the positive side. Preserved so existing callers
        and checkpoints that speak in `K` keep working unchanged.
        """
        m_neg, m_pos = split_from_k(K)
        return cls(
            m_neg=m_neg,
            m_pos=m_pos,
            init_min=init_min,
            init_max=init_max,
            device=device,
        )

    def compile_for_inference(self) -> None:
        """Freeze to a static FP32 LUT and drop the trainable step parameters.

        Export/deployment only: after this call the module has no codebook
        parameters, so it must not be handed to an optimizer.
        """
        cb = self.get_codebook().detach().clone()
        self.compiled_codebook.copy_(cb)
        self.is_compiled = True

        # del, not `= None`: a one-sided codebook already registered the missing
        # side as None, and `del` on a None entry is a no-op on the _parameters
        # dict, so this is correct for both the two-sided and one-sided shapes.
        for name in ("raw_pos_deltas", "raw_neg_deltas"):
            if getattr(self, name, None) is not None:
                delattr(self, name)
                # Re-register as None so attribute access keeps working (and
                # returns None) for callers that introspect the split.
                self.register_parameter(name, None)


# Public alias matching the PRD section 4 class name. The asymmetric split
# codebook (M_neg + 1 + M_pos = K) is the only learned codebook in this
# codebase; the name is kept stable so PRD-named imports resolve.
AsymmetricLearnedCodebook = MemoryEfficientLearnedCodebook
