"""LC-QAT: learned codebook quantization-aware training."""

from nanochat.lcqat.codebook import (
    AsymmetricLearnedCodebook,
    MemoryEfficientLearnedCodebook,
    QuantizedOutput,
)
from nanochat.lcqat.efqat import BlockLatchFreezer, SelectiveFreezer
from nanochat.lcqat.export import (
    attach_learnable_activation_luts,
    export_lcqat_checkpoint,
    wire_activation_luts,
)
from nanochat.lcqat.kd import (
    DenoiserDistiller,
    KDLoss,
    denoiser_kd_loss,
    kd_loss,
)
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
from nanochat.lcqat.sparseprop import (
    SparsePropLinear,
    SparsePropLinearFunction,
    SparsePropLinearLCQAT,
    apply_static_sparsity_mask,
    inject_sparseprop_layers,
)
from nanochat.lcqat.w6 import (
    add_w6_args,
    build_denoiser_teacher,
    build_float_twin,
    describe_sigma_codebooks,
    install_sigma_codebooks,
    latch_threshold,
    make_latch_freezer,
    parse_latch_targets,
    strip_lcqat,
)

__all__ = [
    "AsymmetricLearnedCodebook",
    "MemoryEfficientLearnedCodebook",
    "QuantizedOutput",
    "LCQATLinear",
    "SparsePropLinear",
    "SparsePropLinearLCQAT",
    "SparsePropLinearFunction",
    "apply_static_sparsity_mask",
    "inject_sparseprop_layers",
    "LayerKConfig",
    "PRESETS",
    "SelectiveFreezer",
    "BlockLatchFreezer",
    "ACTIVATION_LUTS",
    "compile_activation",
    "compile_activation_lut",
    "get_activation",
    "kd_loss",
    "KDLoss",
    "denoiser_kd_loss",
    "DenoiserDistiller",
    "export_lcqat_checkpoint",
    "attach_learnable_activation_luts",
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
    "add_w6_args",
    "build_denoiser_teacher",
    "build_float_twin",
    "describe_sigma_codebooks",
    "install_sigma_codebooks",
    "latch_threshold",
    "make_latch_freezer",
    "parse_latch_targets",
    "strip_lcqat",
]
