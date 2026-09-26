"""
Asymmetric learned codebook quantization (LC-QAT PRD sections 2 and 3.1).

Pure tensor math: no training loops, no hardware logic, no config objects.
The parameter names `raw_pos_deltas` / `raw_neg_deltas` are a stable contract:
GPT.setup_optimizer splits them into their own optimizer group by name.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizedOutput:
    """Typed output contract for a quantization step (PRD 2.2)."""

    value: torch.Tensor  # STE-quantized tensor in the input dtype
    indices: torch.Tensor  # codebook indices, uint8 (K <= 255) or int32
    codebook: torch.Tensor  # FP32 codebook of shape (K,) used for this quantization


class MemoryEfficientLearnedCodebook(nn.Module):
    """
    Parametric asymmetric learned codebook for odd cardinality K = 2M + 1.

    Index M = K // 2 is anchored strictly to FP32 0.0. Levels are cumulative
    softplus step sizes, so gradient updates cannot break strict monotonicity
    (PRD 1.1). Forward emits the dequantized value with a straight-through
    estimator that routes gradients to both the input and the codebook (PRD 1.3).
    """

    def __init__(
        self, K: int = 255, init_min: float = -1.0, init_max: float = 1.0, device=None
    ):
        super().__init__()
        if K % 2 != 1 or K < 3:
            raise ValueError(f"Codebook size K must be an odd integer >= 3, got {K}")
        self.K = int(K)
        self.m = self.K // 2  # step count per branch

        # Floor the init span: a zero span (e.g. zero-initialized weights) would
        # make inverse softplus diverge to -inf and freeze the codebook at 0.0.
        span_pos = max(abs(init_max), 1e-6)
        span_neg = max(abs(init_min), 1e-6)

        init_pos = torch.linspace(0, span_pos, self.m + 1, device=device)[1:]
        init_neg = torch.linspace(0, span_neg, self.m + 1, device=device)[1:]
        pos_deltas = init_pos - torch.cat([init_pos.new_zeros(1), init_pos[:-1]])
        neg_deltas = init_neg - torch.cat([init_neg.new_zeros(1), init_neg[:-1]])

        # Inverse softplus parameterization: rho = log(exp(delta) - 1)
        raw_pos = torch.log(torch.expm1(pos_deltas))
        raw_neg = torch.log(torch.expm1(neg_deltas))
        if not (torch.isfinite(raw_pos).all() and torch.isfinite(raw_neg).all()):
            raise ValueError(
                f"Init range too large for softplus inverse: init_min={init_min}, init_max={init_max}"
            )
        self.raw_pos_deltas = nn.Parameter(raw_pos)
        self.raw_neg_deltas = nn.Parameter(raw_neg)

        # Static FP32 LUT for deployment; only read after compile_for_inference()
        self.register_buffer(
            "compiled_codebook",
            torch.empty(self.K, dtype=torch.float32, device=device),
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
        pos_steps = F.softplus(self.raw_pos_deltas)
        neg_steps = F.softplus(self.raw_neg_deltas)
        pos_side = torch.cumsum(pos_steps, dim=0)
        neg_side = -torch.cumsum(neg_steps, dim=0)
        zero = pos_steps.new_zeros(1)
        return torch.cat([neg_side.flip(0), zero, pos_side])

    def forward(self, x: torch.Tensor) -> QuantizedOutput:
        codebook = self.get_codebook().to(torch.float32)
        x_fp32 = x.to(torch.float32)

        midpoints = (codebook[:-1] + codebook[1:]) * 0.5
        indices = torch.bucketize(x_fp32.detach(), midpoints)
        x_dequant = codebook[indices]

        # STE (PRD 1.3): forward value is exactly C[Q]; the zero-valued identity
        # term routes dL/dx, and the live gather routes dL/dC = scatter of dL/dy
        # over assigned buckets. A plain `x + (C[Q] - x).detach()` would zero out
        # the codebook gradient.
        x_q = x_dequant + (x_fp32 - x_fp32.detach())

        idx_dtype = torch.uint8 if self.K <= 255 else torch.int32
        return QuantizedOutput(
            value=x_q.to(x.dtype),
            indices=indices.to(idx_dtype),
            codebook=codebook,
        )

    def compile_for_inference(self) -> None:
        """Freeze to a static FP32 LUT and drop the trainable step parameters.

        Export/deployment only: after this call the module has no codebook
        parameters, so it must not be handed to an optimizer.
        """
        cb = self.get_codebook().detach().clone()
        self.compiled_codebook.copy_(cb)
        self.is_compiled = True

        if hasattr(self, "raw_pos_deltas"):
            del self.raw_pos_deltas
        if hasattr(self, "raw_neg_deltas"):
            del self.raw_neg_deltas


# Public alias matching the PRD section 4 class name. The asymmetric split
# codebook (M_neg + 1 + M_pos = K) is the only learned codebook in this
# codebase; the name is kept stable so PRD-named imports resolve.
AsymmetricLearnedCodebook = MemoryEfficientLearnedCodebook
