"""LC-QAT: learned codebook quantization-aware training."""

from nanochat.lcqat.codebook import MemoryEfficientLearnedCodebook, QuantizedOutput
from nanochat.lcqat.export import export_lcqat_checkpoint, wire_activation_luts
from nanochat.lcqat.linear import LCQATLinear
from nanochat.lcqat.lut import compile_activation_lut
from nanochat.lcqat.retrofit import (
    PRESETS,
    LayerKConfig,
    finish_lcqat_after_load,
    get_layer_config,
    is_exported_lcqat_state,
    is_lcqat_state,
    lcqat_config_from_args,
    parse_k_map,
    prepare_lcqat_before_load,
    retrofit_model,
    retrofit_summary,
)

__all__ = [
    "MemoryEfficientLearnedCodebook",
    "QuantizedOutput",
    "LCQATLinear",
    "LayerKConfig",
    "PRESETS",
    "compile_activation_lut",
    "export_lcqat_checkpoint",
    "wire_activation_luts",
    "finish_lcqat_after_load",
    "get_layer_config",
    "is_exported_lcqat_state",
    "is_lcqat_state",
    "lcqat_config_from_args",
    "parse_k_map",
    "prepare_lcqat_before_load",
    "retrofit_model",
    "retrofit_summary",
]
