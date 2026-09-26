"""LC-QAT: learned codebook quantization-aware training."""

from nanochat.lcqat.codebook import (
    AsymmetricLearnedCodebook,
    MemoryEfficientLearnedCodebook,
    QuantizedOutput,
)
from nanochat.lcqat.efqat import SelectiveFreezer
from nanochat.lcqat.export import export_lcqat_checkpoint, wire_activation_luts
from nanochat.lcqat.kd import KDLoss, kd_loss
from nanochat.lcqat.linear import LCQATLinear
from nanochat.lcqat.lut import (
    ACTIVATION_LUTS,
    compile_activation,
    compile_activation_lut,
    get_activation,
)
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
    "AsymmetricLearnedCodebook",
    "MemoryEfficientLearnedCodebook",
    "QuantizedOutput",
    "LCQATLinear",
    "LayerKConfig",
    "PRESETS",
    "SelectiveFreezer",
    "ACTIVATION_LUTS",
    "compile_activation",
    "compile_activation_lut",
    "get_activation",
    "kd_loss",
    "KDLoss",
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
