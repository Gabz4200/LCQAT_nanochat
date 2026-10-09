"""Runtime 4-bit weight storage with quantization-aware training.

LC-QAT (:mod:`nanochat.models.quant.linear`) quantizes weights through a
*learned* codebook in the forward pass but keeps fp32 master weights for
the optimizer. This module is the fixed-grid counterpart: a linear layer
whose **persistent storage** is 4 bits -- packed nibbles (one uint8 holds
two 4-bit values) plus a single fp32 scale per tensor -- so the stored
weight costs 0.5 bytes/weight instead of 4.

The forward is quantization-aware: it dequantizes the packed nibbles
(``(index - 8) * scale``) and matmuls, with a straight-through estimator
that routes the output gradient to the dequantized weight and the scale.
The fp32 ``weight`` parameter is the optimization surrogate the MinExpLSTM
optimizers step; the packed buffer is the 4-bit storage mirror that
:class:`~nanochat.training.minexp_lstm.SGD_MinExpLstm_Optimizer` and
:class:`~nanochat.training.mezo_minexplstm.MeZo_MinExpLstm_Optimizer`
refresh after every step (Quantization-Aware Training on the 4-bit layout:
dequantize -> step -> requantize), so the weight's persistent form never
leaves 4 bits between steps.

Why an fp32 surrogate: a differentiable optimizer must step a continuous
tensor, and packed nibbles are discrete. The master weight is the standard
QAT surrogate (the same role the learned codebook's raw deltas play in
LC-QAT); the 4-bit packed buffer + scale is the persistent artifact the
PRD's storage pillar asks for, and the INT8 optimizer states (3 bytes/
param) plus 8-bit SGD gradients are where the training-time memory is
actually cut.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from nanochat.models.quant.packing import pack_nibbles, unpack_nibbles


def _quantize_to_nibbles(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize an fp32 weight to 4-bit nibble indices + a per-tensor scale.

    The 16 nibble levels are symmetric around zero: index ``i`` maps to the
    dequantized value ``(i - 8) * scale`` for ``i in 0..15``, so index 8 is
    exactly 0.0 (a structural zero) and the representable range is
    ``[-8 * scale, 7 * scale]``. The scale is the max-abs weight divided by
    8, the symmetric-max-abs convention the INT8 optimizer states use (one
    scale per tensor, no blocking overhead on CPU).
    """
    scale = weight.detach().abs().max() / 8.0
    if not torch.isfinite(scale) or float(scale) <= 0.0:
        scale = weight.detach().flatten()[0].new_tensor(1e-6)
    # Map [-8*scale, 7*scale] onto indices 0..15: index = round(w/scale) + 8.
    indices = torch.clamp(torch.round(weight.detach() / scale) + 8, 0, 15).to(
        torch.uint8
    )
    return indices, scale


def pack_fourbit_weights(indices: Tensor) -> Tensor:
    """Pack ``[..., N]`` nibble indices into ``[..., ceil(N/2)]`` uint8 bytes.

    Thin wrapper over :func:`pack_nibbles` so callers name the 4-bit intent.
    """
    return pack_nibbles(indices)


def unpack_fourbit_weights(packed: Tensor, n: int) -> Tensor:
    """Unpack ``[..., ceil(N/2)]`` bytes back into ``[..., N]`` nibble indices."""
    return unpack_nibbles(packed, n)


def dequantize_fourbit(packed: Tensor, n: int, scale: Tensor) -> Tensor:
    """Dequantize packed nibbles to an fp32 weight of shape ``[..., n]``.

    Index ``i`` dequantizes to ``(i - 8) * scale``; the zero anchor (index 8)
    keeps pruned weights structurally zero.
    """
    indices = unpack_fourbit_weights(packed, n).to(torch.float32)
    return (indices - 8.0) * scale


class FourBitLinear(torch.nn.Module):
    """A linear layer whose weights are stored at 4 bits.

    ``weight`` is the fp32 optimization surrogate (the tensor the
    MinExpLSTM optimizers step); ``weight_packed`` (uint8, ``ceil(N/2)``
    bytes) and ``weight_scale`` (one fp32 scalar) are the persistent
    4-bit storage. ``forward`` runs the quantization-aware matmul
    (dequantize the packed nibbles, STE to the surrogate and the scale),
    so training is aware of the 4-bit grid while the stored weight stays
    at 0.5 bytes/weight.

    The QAT optimizers refresh the mirror after each step via
    :meth:`quantize_weight_`, keeping the persistent form in 4 bits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: tuple[float, float] = (-0.1, 0.1),
        device=None,
        dtype=None,  # noqa: ARG002 - matches nn.Linear's signature
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # fp32 optimization surrogate: the tensor the optimizers step.
        self.weight = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=torch.float32,
            )
        )
        torch.nn.init.uniform_(self.weight, *weight_init)
        # Persistent 4-bit storage: packed nibbles + one fp32 scale.
        indices, scale = _quantize_to_nibbles(self.weight.detach())
        self.weight_packed = pack_fourbit_weights(indices)
        self.weight_scale = scale.clone().detach()
        # Back-reference so the QAT optimizers can refresh the 4-bit mirror.
        self.weight._fourbit_linear = self  # type: ignore[attr-defined]
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float32)
            )
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        # Quantization-aware forward; the STE is inside _quantized_weight.
        w_q = self._quantized_weight()
        return F.linear(input, w_q, self.bias)

    def _quantized_weight(self) -> Tensor:
        """The STE-quantized weight: dequantize(pack(round(w))).

        The packing is detached (the indices are constants in the
        backward); the gradient flows to the fp32 surrogate through the
        dequantized value, the standard fake-quant / STE contract.
        """
        indices, scale = _quantize_to_nibbles(self.weight)
        packed = pack_fourbit_weights(indices)
        w = dequantize_fourbit(packed, self.in_features, scale)
        # STE: forward uses the quantized value, backward passes gradient unchanged.
        return self.weight + (w - self.weight).detach()

    def quantize_weight_(self) -> None:
        """Refresh the 4-bit storage mirror from the fp32 surrogate.

        Called by the QAT optimizers after a step: the optimizer steps
        the fp32 surrogate, then this requantizes it so the persistent
        ``weight_packed`` / ``weight_scale`` stay in 4 bits between
        steps.
        """
        indices, scale = _quantize_to_nibbles(self.weight.detach())
        self.weight_packed = pack_fourbit_weights(indices)
        self.weight_scale = scale.clone().detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, storage=4bit"
        )
