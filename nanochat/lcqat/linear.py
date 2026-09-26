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
        # Backend for the quantized-inference index path (active only once
        # export/load has installed packed_weight_indices buffers).
        self.matmul_backend = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q: QuantizedOutput = self.act_quantizer(x)
        if "packed_weight_indices" in self._buffers:
            return self._forward_quantized(x, x_q)
        w_q: QuantizedOutput = self.weight_quantizer(self.weight)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        y = F.linear(x_q.value, w_q.value.to(dtype=x.dtype), bias)
        if self.out_quantizer is not None:
            y = self.out_quantizer(y).value
        return y

    def _forward_quantized(self, x: torch.Tensor, x_q: QuantizedOutput) -> torch.Tensor:
        """Inference path: fetch weights through packed IDs + LUT (no fp32 weight).

        Activations quantize to IDs as usual; the matmul resolves both sides
        through their codebooks at fetch time. Requires the buffers installed
        by export_lcqat_checkpoint (packed_weight_indices + weight_index_format).
        """
        from nanochat.lcqat.ops import dispatch_index_linear

        leading = x.shape[:-1]
        act_ids = x_q.indices.reshape(-1, self.in_features)
        y = dispatch_index_linear(
            act_ids,
            x_q.codebook,
            self.packed_weight_indices,
            self.weight_quantizer.get_codebook(),
            self.in_features,
            int(self.weight_index_format),
            backend=self.matmul_backend,
        )
        y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        if self.out_quantizer is not None:
            y = self.out_quantizer(y).value
        return y

    def quantized_mlp_chain(
        self, x: torch.Tensor, next_linear: "LCQATLinear"
    ) -> torch.Tensor:
        """Fused quantized MLP chain (PRD 6): c_fc index matmul -> out IDs
        -> relu^2 `activation_lut` table -> c_proj index matmul fed with
        pre-quantized IDs. Float math is only the two matmul accumulations;
        the elementwise op is an index gather.

        Self is c_fc, `next_linear` is c_proj. Requires the buffers
        installed by export (packed weights + activation_lut on self).
        """
        from nanochat.lcqat.ops import dispatch_index_linear

        if self.out_quantizer is None:
            raise RuntimeError("fused MLP chain requires c_fc.out_quantizer")
        if "activation_lut" not in self._buffers:
            raise RuntimeError("fused MLP chain requires the activation_lut buffer")
        if "packed_weight_indices" not in next_linear._buffers:
            raise RuntimeError("fused MLP chain requires packed c_proj weights")
        if self.activation_lut.dtype != torch.uint8:
            raise ValueError(
                f"fused MLP chain supports K_act <= 255 (uint8 table), got "
                f"{self.activation_lut.dtype}"
            )

        leading = x.shape[:-1]
        x_q = self.act_quantizer(x)
        y = dispatch_index_linear(
            x_q.indices.reshape(-1, self.in_features),
            x_q.codebook,
            self.packed_weight_indices,
            self.weight_quantizer.get_codebook(),
            self.in_features,
            int(self.weight_index_format),
            backend=self.matmul_backend,
        )
        y_ids = self.out_quantizer(y).indices.reshape(-1, self.out_features)
        z_ids = self.activation_lut[y_ids.long()]
        out = dispatch_index_linear(
            z_ids.reshape(-1, next_linear.in_features),
            next_linear.act_quantizer.get_codebook(),
            next_linear.packed_weight_indices,
            next_linear.weight_quantizer.get_codebook(),
            next_linear.in_features,
            int(next_linear.weight_index_format),
            backend=self.matmul_backend,
        )
        out = out.reshape(*leading, next_linear.out_features)
        if next_linear.bias is not None:
            out = out + next_linear.bias.to(dtype=out.dtype)
        return out

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
