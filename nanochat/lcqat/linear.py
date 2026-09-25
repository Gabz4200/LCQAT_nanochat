"""
Dual-quantized drop-in Linear (LC-QAT PRD sections 3.2 and 6).

Subclasses nanochat's Linear so every existing structural contract keeps
holding (isinstance checks in num_matmul_params / init_weights / fp8 filters,
`.weight` attribute access), while forward executes the PRD dual-quantization
path: quantized input activations x quantized weights, optionally quantizing
the output as well (Q/K/V projections and MLP c_fc, per PRD training spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear
from nanochat.lcqat.codebook import MemoryEfficientLearnedCodebook, QuantizedOutput


class LCQATLinear(Linear):
    """Drop-in replacement for Linear with configurable odd codebooks.

    Weight and activation quantizers are always present; `quantize_out` adds a
    third codebook on the output (PRD: training must quantize the output of the
    Q/K/V final projections; the MLP c_fc output quantizer is the input side of
    the fused activation LUT, PRD 6).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        K_weight: int = 3,
        K_act: int = 15,
        quantize_out: bool = False,
        out_k: int | None = None,
        weight_init: tuple[float, float] = (-0.1, 0.1),
        act_init: tuple[float, float] = (-2.0, 2.0),
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.K_weight = int(K_weight)
        self.K_act = int(K_act)
        self.weight_quantizer = MemoryEfficientLearnedCodebook(
            K=self.K_weight,
            init_min=weight_init[0],
            init_max=weight_init[1],
            device=device,
        )
        self.act_quantizer = MemoryEfficientLearnedCodebook(
            K=self.K_act, init_min=act_init[0], init_max=act_init[1], device=device
        )
        out_k = self.K_act if out_k is None else int(out_k)
        self.out_quantizer = (
            MemoryEfficientLearnedCodebook(
                K=out_k, init_min=act_init[0], init_max=act_init[1], device=device
            )
            if quantize_out
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q: QuantizedOutput = self.act_quantizer(x)
        w_q: QuantizedOutput = self.weight_quantizer(self.weight)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        y = F.linear(x_q.value, w_q.value.to(dtype=x.dtype), bias)
        if self.out_quantizer is not None:
            y = self.out_quantizer(y).value
        return y

    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        K_weight: int = 3,
        K_act: int = 15,
        quantize_out: bool = False,
        out_k: int | None = None,
        act_init: tuple[float, float] = (-2.0, 2.0),
    ) -> "LCQATLinear":
        if isinstance(mod, LCQATLinear):
            raise ValueError(
                "from_float expects a plain float Linear, got an LCQATLinear"
            )
        if mod.weight.is_meta:
            raise RuntimeError(
                "LCQATLinear.from_float requires materialized weights; "
                "run to_empty()/init_weights() before retrofitting"
            )
        # Weight quantizer spans the observed weight range; a zero span (zero-init
        # projections) is floored inside the codebook so the zero anchor stays exact.
        w_max = mod.weight.detach().abs().max().item()
        new_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=(mod.bias is not None),
            K_weight=K_weight,
            K_act=K_act,
            quantize_out=quantize_out,
            out_k=out_k,
            weight_init=(-w_max, w_max),
            act_init=act_init,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        with torch.no_grad():
            new_mod.weight.copy_(mod.weight)
            if mod.bias is not None:
                new_mod.bias.copy_(mod.bias)
        return new_mod
