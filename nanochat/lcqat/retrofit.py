"""
Per-layer LC-QAT retrofitting (LC-QAT PRD section 4).

Maps nanochat module names to (K_weight, K_act) allocations, replaces plain
nn.Linear modules with LCQATLinear in place, and provides the checkpoint
detection helpers shared by scripts/base_train.py and checkpoint_manager.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch.nn as nn

from nanochat.lcqat.linear import LCQATLinear


@dataclass(frozen=True)
class LayerKConfig:
    """Codebook cardinalities per module role (all odd, >= 3).

    Roles are nanochat module-name suffixes: attn.c_q/c_k, attn.c_v,
    attn.c_proj (o_proj), mlp.c_fc (gate/up), mlp.c_proj (down_proj).
    `k_map` holds explicit (name_substring, K_weight, K_act) overrides,
    checked in order before the role defaults.
    """

    qk_weight: int = 3
    qk_act: int = 15
    v_weight: int = 15
    v_act: int = 15
    o_weight: int = 15
    o_act: int = 15
    fc_weight: int = 15
    fc_act: int = 15
    down_weight: int = 15
    down_act: int = 15
    quantize_qkv_out: bool = True
    quantize_fc_out: bool = True
    k_map: tuple[tuple[str, int, int], ...] = ()
    min_linear_dim: int = 128

    @classmethod
    def from_dict(cls, data: dict) -> "LayerKConfig":
        kwargs = dict(data)
        if "k_map" in kwargs:
            kwargs["k_map"] = tuple(
                (str(s), int(kw), int(ka)) for s, kw, ka in kwargs["k_map"]
            )
        known = {f.name for f in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(f"Unknown LayerKConfig keys: {sorted(unknown)}")
        return cls(**kwargs)

    def validate(self) -> None:
        k_fields = {
            name: value
            for name, value in vars(self).items()
            if name.endswith(("_weight", "_act"))
        }
        for name, value in k_fields.items():
            if value % 2 != 1 or value < 3:
                raise ValueError(
                    f"LayerKConfig.{name} must be an odd integer >= 3, got {value}"
                )
        for substr, kw, ka in self.k_map:
            for label, value in (("K_weight", kw), ("K_act", ka)):
                if value % 2 != 1 or value < 3:
                    raise ValueError(
                        f"k_map rule {substr!r} {label} must be an odd integer >= 3, got {value}"
                    )


# small: maximum-compression default (user-approved). prd: PRD section 4 table
# verbatim (8-bit down_proj / residual recombination path).
PRESETS: dict[str, LayerKConfig] = {
    "small": LayerKConfig(),
    "prd": LayerKConfig(down_weight=255, down_act=255),
}


def parse_k_map(spec: str) -> tuple[tuple[str, int, int], ...]:
    """Parse `substr:KW/KA,substr:KW/KA` into override rules."""
    rules: list[tuple[str, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part or "/" not in part.split(":", 1)[1]:
            raise ValueError(f"k-map entry {part!r} must have the form substring:KW/KA")
        substr, ks = part.rsplit(":", 1)
        kw_s, ka_s = ks.split("/", 1)
        try:
            kw, ka = int(kw_s), int(ka_s)
        except ValueError as error:
            raise ValueError(
                f"k-map entry {part!r} has non-integer K values"
            ) from error
        for label, value in (("K_weight", kw), ("K_act", ka)):
            if value % 2 != 1 or value < 3:
                raise ValueError(
                    f"k-map entry {part!r}: {label} must be an odd integer >= 3, got {value}"
                )
        if not substr.strip():
            raise ValueError(f"k-map entry {part!r} has an empty substring")
        rules.append((substr.strip(), kw, ka))
    if not rules:
        raise ValueError(f"k-map spec {spec!r} contains no rules")
    return tuple(rules)


def lcqat_config_from_args(args) -> LayerKConfig:
    """Build the LayerKConfig from --lcqat-preset / --lcqat-k-map flags."""
    if args.lcqat_preset not in PRESETS:
        raise ValueError(
            f"Unknown --lcqat-preset {args.lcqat_preset!r}, valid: {sorted(PRESETS)}"
        )
    cfg = PRESETS[args.lcqat_preset]
    if args.lcqat_k_map:
        cfg = replace(cfg, k_map=parse_k_map(args.lcqat_k_map))
    return cfg


# Module-name roles: (K_weight field, K_act field, quantize output?)
_ROLE_QK = ("qk_weight", "qk_act", "quantize_qkv_out")
_ROLE_V = ("v_weight", "v_act", "quantize_qkv_out")
_ROLE_O = ("o_weight", "o_act", None)
_ROLE_FC = ("fc_weight", "fc_act", "quantize_fc_out")
_ROLE_DOWN = ("down_weight", "down_act", None)


def get_layer_config(
    module_name: str, config: LayerKConfig
) -> tuple[int, int, bool] | None:
    """Return (K_weight, K_act, quantize_out) for a module, or None to skip.

    Matching is on nanochat role suffixes, not the PRD's q_proj/down_proj
    names: attn.c_q/c_k are q/k, mlp.c_proj is down_proj, attn.c_proj is
    o_proj. Unmatched names (lm_head, ve_gate, smear_gate) are skipped.
    """
    role = None
    if "attn.c_q" in module_name or "attn.c_k" in module_name:
        role = _ROLE_QK
    elif "attn.c_v" in module_name:
        role = _ROLE_V
    elif "attn.c_proj" in module_name:
        role = _ROLE_O
    elif "mlp.c_fc" in module_name:
        role = _ROLE_FC
    elif "mlp.c_proj" in module_name:
        role = _ROLE_DOWN
    if role is None:
        return None

    k_weight_field, k_act_field, out_field = role
    k_weight, k_act = getattr(config, k_weight_field), getattr(config, k_act_field)
    for substr, kw, ka in config.k_map:
        if substr in module_name:
            k_weight, k_act = kw, ka
            break
    quantize_out = bool(getattr(config, out_field)) if out_field is not None else False
    return k_weight, k_act, quantize_out


def retrofit_model(model: nn.Module, config: LayerKConfig) -> nn.Module:
    """Recursively replace eligible nn.Linear modules with LCQATLinear, in place.

    Skips lm_head and any module below `min_linear_dim` (mirrors the FP8
    module filter), skips already-retrofitted modules (idempotent), and
    requires materialized (non-meta) weights.
    """
    config.validate()
    replacements: list[tuple[nn.Module, str, nn.Linear, tuple[int, int, bool]]] = []
    matched_rules: set[str] = set()

    for full_name, child in model.named_modules():
        if not isinstance(child, nn.Linear) or isinstance(child, LCQATLinear):
            continue
        spec = get_layer_config(full_name, config)
        if spec is None:
            continue
        if "Float8" in type(child).__name__:
            raise ValueError(
                f"--fp8 and --lcqat are mutually exclusive (both convert Linear): {full_name}"
            )
        if min(child.in_features, child.out_features) < config.min_linear_dim:
            continue
        if child.weight.is_meta:
            raise RuntimeError(
                f"Cannot retrofit {full_name}: weight is on the meta device; "
                "run to_empty()/init_weights() first"
            )
        for substr, _, _ in config.k_map:
            if substr in full_name:
                matched_rules.add(substr)
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, attr, child, spec))

    unmatched = {substr for substr, _, _ in config.k_map} - matched_rules
    if unmatched:
        raise ValueError(f"--lcqat-k-map rules matched no modules: {sorted(unmatched)}")

    for parent, attr, child, (k_weight, k_act, quantize_out) in replacements:
        setattr(
            parent,
            attr,
            LCQATLinear.from_float(
                child,
                K_weight=k_weight,
                K_act=k_act,
                quantize_out=quantize_out,
                out_k=k_act,
            ),
        )
    return model


def retrofit_summary(model: nn.Module) -> dict[str, int]:
    """Count retrofitted modules per (K_weight, K_act) for logging."""
    summary: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, LCQATLinear):
            key = f"Kw={module.K_weight},Ka={module.K_act}"
            summary[key] = summary.get(key, 0) + 1
    return summary


def is_lcqat_state(state_dict: dict) -> bool:
    """True if a checkpoint state_dict was trained with LC-QAT."""
    return any(key.endswith("raw_pos_deltas") for key in state_dict)


def is_exported_lcqat_state(state_dict: dict) -> bool:
    """True if a checkpoint is an exported LC-QAT artifact (weights stripped)."""
    return any(key.endswith("packed_weight_indices") for key in state_dict)


def resolve_lcqat_config(
    meta_lcqat: dict | None, requested: LayerKConfig | None
) -> LayerKConfig:
    """Pick the active config: checkpoint meta (provenance) > requested flags > default."""
    if meta_lcqat:
        return LayerKConfig.from_dict(meta_lcqat)
    if requested is not None:
        return requested
    return PRESETS["small"]


def _prepare_exported_buffers(model: nn.Module, model_data: dict) -> None:
    """Strip shadow weights and register the exported runtime buffers.

    Turns a retrofitted float-structure model into the shape the exported
    artifact expects: each LCQATLinear loses `weight` and gains
    `packed_weight_indices`, `weight_index_format` (and `activation_lut`
    when present), taken from the state itself so shapes and formats can
    never disagree. A missing index key means the active LayerKConfig does
    not match the artifact - fail fast with that hint.
    """
    for name, module in model.named_modules():
        if not isinstance(module, LCQATLinear):
            continue
        idx_key = f"{name}.packed_weight_indices"
        if idx_key not in model_data:
            raise RuntimeError(
                f"exported artifact has no {idx_key}: the active LC-QAT config "
                "does not match the model structure this artifact was built from"
            )
        # Freeze quantizers exactly like export does: raw step params are
        # deleted, only compiled_codebook buffers remain in the state.
        module.weight_quantizer.compile_for_inference()
        module.act_quantizer.compile_for_inference()
        if module.out_quantizer is not None:
            module.out_quantizer.compile_for_inference()
        del module.weight
        module.register_buffer("packed_weight_indices", model_data[idx_key])
        module.register_buffer(
            "weight_index_format", model_data[f"{name}.weight_index_format"]
        )
        lut_key = f"{name}.activation_lut"
        if lut_key in model_data:
            module.register_buffer("activation_lut", model_data[lut_key])


def prepare_lcqat_before_load(
    model: nn.Module,
    model_data: dict,
    meta_lcqat: dict | None,
    requested: LayerKConfig | None,
) -> LayerKConfig | None:
    """Prepare a model for load_state_dict when the checkpoint is LC-QAT.

    Returns the active config, or None when the checkpoint is plain float
    (caller may then retrofit after loading via finish_lcqat_after_load).

    Exported artifacts (weights stripped) are supported for inference: the
    model is retrofitted to the checkpoint's config and re-shaped into the
    exported runtime structure (packed IDs, no shadow weights); training
    from such a state is rejected by the caller (checkpoint_manager).
    """
    if is_exported_lcqat_state(model_data):
        if requested is not None:
            raise RuntimeError(
                "cannot start QAT training from an exported LC-QAT artifact "
                "(weights are stripped); pass the artifact's config via meta "
                "or load a training checkpoint instead"
            )
        config = resolve_lcqat_config(meta_lcqat, None)
        retrofit_model(model, config)
        _prepare_exported_buffers(model, model_data)
        return config
    if is_lcqat_state(model_data):
        config = resolve_lcqat_config(meta_lcqat, requested)
        retrofit_model(model, config)
        return config
    return None


def finish_lcqat_after_load(
    model: nn.Module, requested: LayerKConfig | None
) -> LayerKConfig | None:
    """Retrofit a freshly loaded float checkpoint when QAT start was requested."""
    if requested is None:
        return None
    retrofit_model(model, requested)
    return requested
