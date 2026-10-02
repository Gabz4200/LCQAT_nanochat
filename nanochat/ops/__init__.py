"""LC-QAT op dispatcher and public op facade."""

from nanochat.ops.dispatch import (
    dispatch_gemv,
    dispatch_index_linear,
    dispatch_quant_attn,
)

__all__ = ["dispatch_gemv", "dispatch_index_linear", "dispatch_quant_attn"]
