"""
Per-layer LC-QAT retrofitting (LC-QAT PRD section 4).

Maps nanochat module names to (K_weight, K_act) allocations, replaces plain
nn.Linear modules with LCQATLinear in place, and provides the checkpoint
detection helpers shared by scripts/base_train.py and checkpoint_manager.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, fields, replace

import torch
import torch.nn as nn

from nanochat.models.io import CodebookSpec, LayerQuantSpec
from nanochat.models.quant.activation import ACT_BODIES, ACT_BODY_PWL
from nanochat.models.quant.codebook import split_from_k
from nanochat.models.quant.codebook import validate_split as _validate_split
from nanochat.models.quant.learnable_lut import RELAXATION_LOGITS, RELAXATIONS
from nanochat.models.quant.linear import (
    GRAD_SCALE_INV_SQRT_N,
    GRAD_SCALES,
    LCQATLinear,
    is_lcqat_layer,
)

#: The exact set of `LayerKConfig` fields whose value is a `CodebookSpec`.
#:
#: An explicit list, not a suffix heuristic (`name.endswith(("_weight", "_act"))`).
#: The heuristic cannot work here: `per_channel_weight` is a `bool` whose name
#: ends in `_weight`, so it is fed to `_validate_codebook_spec` as a codebook
#: spec. A bool IS an int in Python, and `False <= 3`, so it raises. Naming the
#: fields makes that class of mistake impossible; a suffix rule would also break
#: every existing checkpoint's `meta["lcqat"]` if the field were ever renamed.
CODEBOOK_SPEC_FIELDS: frozenset[str] = frozenset(
    {
        "qk_weight",
        "qk_act",
        "v_weight",
        "v_act",
        "o_weight",
        "o_act",
        "fc_weight",
        "fc_act",
        "down_weight",
        "down_act",
        "qkv_out",
        "fc_out",
    }
)


def _validate_codebook_spec(spec: CodebookSpec, label: str) -> None:
    """Raise unless `spec` is an int >= 3 or a valid `(m_neg, m_pos)` pair."""
    # A bool is an int subclass, so `per_channel_weight=False` reaches the
    # `isinstance(spec, int)` branch below as a "K of 0" without this guard --
    # and False <= 3 would then raise the *wrong* error for the *right* reason.
    # CODEBOOK_SPEC_FIELDS keeps booleans out of this function entirely; this
    # guard is the defense-in-depth that makes that exclusion a non-issue.
    if isinstance(spec, bool):
        raise ValueError(
            f"{label} is a bool, not a codebook spec; a codebook cardinality is "
            f"an int K or an (m_neg, m_pos) tuple, got {spec!r}"
        )
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(
                f"{label}: an asymmetric split must be (m_neg, m_pos), got {spec!r}"
            )
        m_neg, m_pos = spec
        if (
            isinstance(m_neg, bool)
            or isinstance(m_pos, bool)
            or not isinstance(m_neg, int)
            or not isinstance(m_pos, int)
        ):
            raise ValueError(f"{label}: m_neg/m_pos must be ints, got {spec!r}")
        # Delegate the structural rules (non-negative, one-sided needs >= 2, and
        # K >= 3) to the codebook's own validator so there is one source of truth.
        _validate_split(m_neg, m_pos)
        return
    if isinstance(spec, bool) or not isinstance(spec, int):
        raise ValueError(
            f"{label} must be an int K or an (m_neg, m_pos) tuple, got {spec!r}"
        )
    if spec < 3:
        raise ValueError(f"{label} must be >= 3, got {spec}")


def spec_k(spec: CodebookSpec) -> int:
    """Total level count K implied by a `CodebookSpec`."""
    if isinstance(spec, tuple):
        m_neg, m_pos = spec
        _validate_split(m_neg, m_pos)
        return m_neg + 1 + m_pos
    return int(spec)


def spec_split(spec: CodebookSpec) -> tuple[int, int]:
    """`(m_neg, m_pos)` implied by a `CodebookSpec`, splitting an int K evenly."""
    if isinstance(spec, tuple):
        m_neg, m_pos = spec
        _validate_split(m_neg, m_pos)
        return m_neg, m_pos
    return split_from_k(int(spec))


def _as_spec(value) -> CodebookSpec:
    """Normalize a JSON-decoded spec to int or tuple.

    JSON has no tuples, so a saved `(m_neg, m_pos)` split reads back as a list.
    """
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(
                f"asymmetric split must have 2 entries (m_neg, m_pos), got {value!r}"
            )
        return (int(value[0]), int(value[1]))
    return int(value)


@dataclass(frozen=True)
class LayerKConfig:
    """Codebook cardinalities per module role (each >= 3).

    Roles are nanochat module-name suffixes: attn.c_q/c_k, attn.c_v,
    attn.c_proj (o_proj), mlp.c_fc (gate/up), mlp.c_proj (down_proj).
    `k_map` holds explicit (name_substring, K_weight, K_act) overrides,
    checked in order before the role defaults.

    Each field is a `CodebookSpec`: an int K (symmetric) or an explicit
    `(m_neg, m_pos)` split. The asymmetric form matters because `gpt.py`'s MLP
    computes `relu(x).square()` before `mlp.c_proj`, so the 4*n_embd hidden
    tensor is non-negative: a symmetric codebook puts half its levels on the
    negative side, which that tensor never occupies.
    """

    qk_weight: CodebookSpec = 3
    qk_act: CodebookSpec = 15
    v_weight: CodebookSpec = 15
    v_act: CodebookSpec = 15
    o_weight: CodebookSpec = 15
    o_act: CodebookSpec = 15
    fc_weight: CodebookSpec = 15
    fc_act: CodebookSpec = 15
    down_weight: CodebookSpec = 15
    down_act: CodebookSpec = 15
    # Output quantizers (c_q/c_k/c_v output, c_fc output). `fc_out` is the
    # non-negative MLP hidden -> one-sided by default in the asym preset.
    qkv_out: CodebookSpec = 15
    fc_out: CodebookSpec = 15
    quantize_qkv_out: bool = True
    quantize_fc_out: bool = True
    k_map: tuple[tuple[str, CodebookSpec, CodebookSpec], ...] = ()
    min_linear_dim: int = 128
    # PRD 2.4: "none" (plain STE) or "inv_sqrt_n" (1/sqrt(numel) on the
    # codebook gradient). Config-level so a checkpoint records which one it was
    # trained under, rather than silently changing behaviour on reload.
    grad_scale: str = GRAD_SCALE_INV_SQRT_N
    # PRD 3.4: one weight codebook per output channel instead of one shared.
    # Config-level for the same reason as `grad_scale` -- a checkpoint has to
    # record that it was trained this way, or a reload silently builds shared
    # tables and the saved per-channel parameters are orphaned.
    per_channel_weight: bool = False
    # Quantize the bias through a per-layer learned codebook
    # (`LCQATLinear.bias_quantizer`). Off by default, and deliberately *not* a
    # CODEBOOK_SPEC_FIELDS entry: it is a bool, so `validate()` must not feed it
    # to `_validate_codebook_spec` -- that is exactly the §9.3 Defect 1 class of
    # bug (a bool reaching an int validator) that the explicit frozenset exists
    # to make impossible. Its cardinality, when enabled, defaults to `K_act`.
    #
    # Config-level for the same reason as `grad_scale`: a checkpoint trained
    # with a bias codebook carries its step parameters, so a reload that skipped
    # the flag would fail the strict state_dict load with "unexpected keys".
    quantize_bias: bool = False
    # D9: how the *trained* activation LUT selects its output level.
    # "logits" is the original free `(K_in, K_out)` logit matrix with a
    # straight-through round; "proximity" replaces it with a `knots + levels`
    # pair selected by inverse-square distance, which costs K_in + K_in
    # parameters instead of K_in x K_out and has no softmax-saturation cliff
    # (dev/HANDOFF_symbiosis.md §10.2.1).
    #
    # Config-level for the same reason as `grad_scale`: a checkpoint records
    # which relaxation produced its LUT weights, so a reload does not silently
    # build a different module with orphaned parameters.
    lut_relaxation: str = RELAXATION_LOGITS
    # D9: which body approximates the elementwise activation between two
    # quantized layers. "pwl" is the shipped frozen K_in -> K_out index table
    # (exact at the knots, linear between). "smoothpwl" is the learnable
    # radial-basis body from `activation.py`.
    #
    # Same reasoning as `lut_relaxation`: the two bodies have different
    # parameter shapes, so a checkpoint trained with one cannot be reloaded as
    # the other.
    act_body: str = ACT_BODY_PWL

    @classmethod
    def from_dict(cls, data: dict) -> "LayerKConfig":
        kwargs = dict(data)
        if "k_map" in kwargs:
            # A split survives a JSON round-trip as a list, so normalize to tuple.
            kwargs["k_map"] = tuple(
                (str(s), _as_spec(kw), _as_spec(ka)) for s, kw, ka in kwargs["k_map"]
            )
        for name in sorted(CODEBOOK_SPEC_FIELDS):
            if name in kwargs:
                kwargs[name] = _as_spec(kwargs[name])
        known = {f.name for f in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(f"Unknown LayerKConfig keys: {sorted(unknown)}")
        return cls(**kwargs)

    def validate(self) -> None:
        for substr, kw, ka in self.k_map:
            _validate_codebook_spec(kw, f"k_map rule {substr!r} K_weight")
            _validate_codebook_spec(ka, f"k_map rule {substr!r} K_act")
        # Every codebook-spec field must be a *declared* member of
        # CODEBOOK_SPEC_FIELDS, checked here rather than assumed. A field
        # holding a CodebookSpec that nobody validates is a config that fails
        # at model-build time instead of flag-parse time, which is the same
        # late-failure mode the explicit set exists to prevent.
        undeclared = CODEBOOK_SPEC_FIELDS - {f.name for f in fields(self)}
        if undeclared:
            raise ValueError(
                f"CODEBOOK_SPEC_FIELDS names fields LayerKConfig does not "
                f"define: {sorted(undeclared)}"
            )
        for name in CODEBOOK_SPEC_FIELDS:
            _validate_codebook_spec(getattr(self, name), f"LayerKConfig.{name}")
        if self.grad_scale not in GRAD_SCALES:
            raise ValueError(
                f"LayerKConfig.grad_scale must be one of {GRAD_SCALES}, got "
                f"{self.grad_scale!r}"
            )
        # Both are validated here rather than trusted from argparse, because a
        # config also arrives from `from_dict` on a resume -- and a resume that
        # silently accepted a typo would build a module whose parameter shape
        # does not match the saved weights.
        if self.lut_relaxation not in RELAXATIONS:
            raise ValueError(
                f"LayerKConfig.lut_relaxation must be one of {RELAXATIONS}, got "
                f"{self.lut_relaxation!r}"
            )
        if self.act_body not in ACT_BODIES:
            raise ValueError(
                f"LayerKConfig.act_body must be one of {ACT_BODIES}, got "
                f"{self.act_body!r}"
            )

    def as_dict(self) -> dict:
        """JSON-serializable form (splits become lists, as JSON has no tuples)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["k_map"] = [
            (
                s,
                list(kw) if isinstance(kw, tuple) else kw,
                list(ka) if isinstance(ka, tuple) else ka,
            )
            for s, kw, ka in self.k_map
        ]
        for name in sorted(CODEBOOK_SPEC_FIELDS):
            if isinstance(data[name], tuple):
                data[name] = list(data[name])
        return data


# small: maximum-compression default (user-approved). prd: PRD section 4 table
# verbatim (8-bit down_proj / residual recombination path).
#
# asym (the default): same total level counts as `small`, but split so the two
# non-negative tensors stop wasting half their codebook. `gpt.py`'s MLP is
# `c_fc -> relu(x).square() -> c_proj`, so
#   * `mlp.c_fc`'s OUTPUT codebook sees relu^2 >= 0  -> m_neg = 0
#   * `mlp.c_proj`'s INPUT codebook sees relu^2 >= 0  -> m_neg = 0
# A symmetric 15 spends 7 levels on a sign those tensors never take; 0/7 spends
# all 8. Everything upstream of the activation (attention in/out, c_fc in) is
# genuinely signed and gets an asymmetric split rather than a symmetric one.
PRESETS: dict[str, LayerKConfig] = {
    "small": LayerKConfig(),
    "prd": LayerKConfig(down_weight=255, down_act=255),
    "asym": LayerKConfig(
        # Attention: signed (RMSNorm'd) inputs and outputs, mildly skewed.
        qk_weight=(1, 1),
        qk_act=(6, 8),
        v_weight=(6, 8),
        v_act=(6, 8),
        o_weight=(6, 8),
        o_act=(6, 8),
        # c_fc input is signed (post-RMSNorm); its OUTPUT is relu^2 >= 0.
        fc_weight=(6, 8),
        fc_act=(6, 8),
        # c_proj input is relu^2 >= 0 (one-sided); its output rejoins the stream.
        down_weight=(6, 8),
        down_act=(0, 7),
        qkv_out=(6, 8),
        fc_out=(0, 7),
    ),
}

DEFAULT_PRESET = "asym"


def _parse_spec_token(token: str, label: str) -> CodebookSpec:
    """Parse one K token: `8` or `0-7` (m_neg-m_pos)."""
    token = token.strip()
    if not token:
        raise ValueError(f"{label}: empty codebook spec")
    if "-" in token[1:]:
        neg, _, pos = token.partition("-")
        spec = (int(neg), int(pos))
    else:
        spec = int(token)
    _validate_codebook_spec(spec, label)
    return spec


def parse_k_map(spec: str) -> tuple[tuple[str, CodebookSpec, CodebookSpec], ...]:
    """Parse `substr:KW/KA,substr:KW/KA` into override rules.

    Each K is either a total level count (`15`) or an asymmetric split written
    `m_neg-m_pos` (`0-7`).
    """
    rules: list[tuple[str, CodebookSpec, CodebookSpec]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part or "/" not in part.split(":", 1)[1]:
            raise ValueError(
                f"k-map entry {part!r} must have the form "
                "substring:KW/KA, with each K either N or Mneg-Mpos"
            )
        substr, ks = part.rsplit(":", 1)
        kw_s, ka_s = ks.split("/", 1)
        kw = _parse_spec_token(kw_s, f"k-map entry {part!r} K_weight")
        ka = _parse_spec_token(ka_s, f"k-map entry {part!r} K_act")
        if not substr.strip():
            raise ValueError(f"k-map entry {part!r} has an empty substring")
        rules.append((substr.strip(), kw, ka))
    if not rules:
        raise ValueError(f"k-map spec {spec!r} contains no rules")
    return tuple(rules)


def add_lcqat_args(parser: argparse.ArgumentParser) -> None:
    """Register the `--lcqat-*` / `--codebook-*` flags on `parser`.

    Lives here, next to `lcqat_config_from_args` which reads them, and beside
    `add_sparseprop_pruning_args` / `add_w6_args` which exist for the same
    reason: `base_train`, `chat_sft` and `chat_rl` each spell this block out,
    and a flag added to one but not another would make the three disagree about
    what a run means. Every default here matches the argparse default each
    script had, and `lcqat_config_from_args` reads the rest via `getattr`, so a
    caller that registers a subset still works.

    LC-QAT is ON by default; `--no-lcqat` is the opt-out.
    """
    parser.add_argument(
        "--no-lcqat",
        action="store_false",
        dest="lcqat",
        default=True,
        help="disable LC-QAT (default: LC-QAT is always on; pass this to run plain float training)",
    )
    parser.add_argument(
        "--lcqat-preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS),
        help=(
            "per-layer K allocation. 'asym' (default) splits each codebook by sign, "
            "giving m_neg=0 to the two non-negative MLP tensors so no level is spent "
            "on a sign they never take. 'small' is the symmetric max-compression "
            "table, 'prd' the PRD table."
        ),
    )
    parser.add_argument(
        "--lcqat-k-map",
        type=str,
        default="",
        help="override K per module substring, e.g. 'mlp.c_proj:255/255,attn.c_v:15/15'",
    )
    parser.add_argument(
        "--codebook-lr",
        type=float,
        default=1e-3,
        help="learning rate for codebook step parameters (PRD: 10-50x network weights)",
    )
    parser.add_argument(
        "--codebook-grad-scale",
        type=str,
        default=GRAD_SCALE_INV_SQRT_N,
        choices=list(GRAD_SCALES),
        help=(
            "PRD 2.4 codebook gradient scaling. 'inv_sqrt_n' (default) scales the "
            "codebook gradient by 1/sqrt(numel) -- in a 4096x4096 layer 16.7M "
            "elements pool into one K-entry codebook, and unscaled the step "
            "parameters oscillate relative to the weights. 'none' uses the plain "
            "STE. Note the two interact multiplicatively with --codebook-lr: with "
            "N = B*T*D in the millions the activation codebook gradient is ~1000x "
            "smaller under 'inv_sqrt_n', so this is a real trade, not a free win."
        ),
    )


def lcqat_config_from_args(args) -> LayerKConfig:
    """Build the LayerKConfig from the `--lcqat-*` flags.

    Reads `--lcqat-preset`, `--lcqat-k-map`, `--codebook-grad-scale` and
    `--lcqat-channel-center`. Every one after the preset is read via `getattr`
    with a default, so a caller that registers a subset of the flags still
    works -- `chat_rl` does not register `--lcqat-channel-center` at all.

    `--lcqat-channel-center` sets `per_channel_weight`, and the resulting config
    is what the three training scripts persist to `meta["lcqat"]`, so a resume
    rebuilds per-channel tables instead of silently reverting to shared ones
    and orphaning the saved per-channel
    parameters.
    """
    if args.lcqat_preset not in PRESETS:
        raise ValueError(
            f"Unknown --lcqat-preset {args.lcqat_preset!r}, valid: {sorted(PRESETS)}"
        )
    cfg = PRESETS[args.lcqat_preset]
    if args.lcqat_k_map:
        cfg = replace(cfg, k_map=parse_k_map(args.lcqat_k_map))
    if getattr(args, "lcqat_channel_center", False):
        cfg = replace(cfg, per_channel_weight=True)
    grad_scale = getattr(args, "codebook_grad_scale", None)
    if grad_scale is not None:
        if grad_scale not in GRAD_SCALES:
            raise ValueError(
                f"Unknown --codebook-grad-scale {grad_scale!r}, "
                f"valid: {sorted(GRAD_SCALES)}"
            )
        cfg = replace(cfg, grad_scale=grad_scale)
    # D9: the activation-LUT relaxation and the activation body. Both default to
    # the shipped behaviour, and both are read with a `getattr` default so a
    # caller that registers a subset of the flags keeps working.
    lut_relaxation = getattr(args, "lcqat_lut_relaxation", None)
    if lut_relaxation is not None:
        if lut_relaxation not in RELAXATIONS:
            raise ValueError(
                f"Unknown --lcqat-lut-relaxation {lut_relaxation!r}, "
                f"valid: {sorted(RELAXATIONS)}"
            )
        cfg = replace(cfg, lut_relaxation=lut_relaxation)
    act_body = getattr(args, "lcqat_act_body", None)
    if act_body is not None:
        if act_body not in ACT_BODIES:
            raise ValueError(
                f"Unknown --lcqat-act-body {act_body!r}, valid: {sorted(ACT_BODIES)}"
            )
        cfg = replace(cfg, act_body=act_body)
    cfg.validate()
    return cfg


# Module-name roles: (weight spec field, act spec field, quantize-out flag field,
# out spec field). A `None` flag field means the role has no output quantizer.
_ROLE_QK = ("qk_weight", "qk_act", "quantize_qkv_out", "qkv_out")
_ROLE_V = ("v_weight", "v_act", "quantize_qkv_out", "qkv_out")
_ROLE_O = ("o_weight", "o_act", None, None)
_ROLE_FC = ("fc_weight", "fc_act", "quantize_fc_out", "fc_out")
_ROLE_DOWN = ("down_weight", "down_act", None, None)


def get_layer_config(module_name: str, config: LayerKConfig) -> LayerQuantSpec | None:
    """Return the quantization plan for a module, or None if it is not retrofitted.

    `output` is only meaningful when `quantize_output` is True; the contract
    normalizes the inert case to the activation spec so no unpack site has to
    carry that precondition on its own.
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

    k_weight_field, k_act_field, out_field, out_spec_field = role
    k_weight, k_act = getattr(config, k_weight_field), getattr(config, k_act_field)
    for substr, kw, ka in config.k_map:
        if substr in module_name:
            k_weight, k_act = kw, ka
            break
    if out_field is None:
        return LayerQuantSpec(weight=k_weight, activation=k_act)
    quantize_out = bool(getattr(config, out_field))
    out_spec = getattr(config, out_spec_field) if quantize_out else None
    return LayerQuantSpec(
        weight=k_weight,
        activation=k_act,
        quantize_output=quantize_out,
        output=out_spec,
    )


def _is_fp8_linear(module: nn.Module) -> bool:
    """Whether `module` is an fp8 Linear installed by `--fp8`.

    The precise check first: a real `Float8Linear` is what `--fp8` installs.
    The name test is the fallback, because that is the codebase-wide idiom for
    identifying fp8 layers (`scripts/_train/build.py`'s `num_fp8` count and
    `disable_fp8` both use it), and it catches an fp8 layer whose class came
    from a build that does not expose `Float8Linear` for import.
    """
    from nanochat.models.fp8 import Float8Linear

    return isinstance(module, Float8Linear) or "Float8" in type(module).__name__


def retrofit_model(model: nn.Module, config: LayerKConfig) -> nn.Module:
    """Recursively replace eligible nn.Linear modules with LCQATLinear, in place.

    Skips lm_head and any module below `min_linear_dim` (mirrors the FP8
    module filter), skips already-retrofitted modules (idempotent), and
    requires materialized (non-meta) weights.
    """
    config.validate()
    replacements: list[tuple[nn.Module, str, nn.Linear, LayerQuantSpec]] = []
    matched_rules: set[str] = set()

    for full_name, child in model.named_modules():
        # `is_lcqat_layer` for the already-retrofitted skip: a
        # SparsePropLinearLCQAT is an nn.Linear (through backbone
        # Linear) and would otherwise be quantized a second time. The
        # pipeline retrofits before it wraps, so this changes nothing
        # there -- it removes the trap for any other call order.
        if not isinstance(child, nn.Linear) or is_lcqat_layer(child):
            continue
        spec = get_layer_config(full_name, config)
        if spec is None:
            continue
        if _is_fp8_linear(child):
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

    for parent, attr, child, spec in replacements:
        setattr(
            parent,
            attr,
            LCQATLinear.from_float(
                child,
                K_weight=spec_k(spec.weight),
                K_act=spec_k(spec.activation),
                quantize_out=spec.quantize_output,
                out_k=spec_k(spec.effective_output),
                K_weight_split=spec_split(spec.weight),
                K_act_split=spec_split(spec.activation),
                out_split=spec_split(spec.effective_output),
                grad_scale=config.grad_scale,
                per_channel_weight=config.per_channel_weight,
                # A per-layer no-op where the module has no bias: every nanochat
                # projection is `bias=False`, so this is the normal case, not an
                # error. `from_float` still raises on the direct-construction
                # combination.
                quantize_bias=config.quantize_bias,
            ),
        )
    return model


def retrofit_summary(model: nn.Module) -> dict[str, int]:
    """Count retrofitted modules per (weight split, act split) for logging.

    Splits are logged rather than bare K because the sign split is the thing
    that changes the effective resolution: `m_neg=0, m_pos=7` and
    `m_neg=3, m_pos=4` are both K=8 but only the first spends all 8 levels on a
    non-negative tensor.
    """
    summary: dict[str, int] = {}
    for module in model.modules():
        # `is_lcqat_layer`, not isinstance: a SparseProp-wrapped LC-QAT layer is
        # one without being a subclass, and it carries both quantizers.
        if is_lcqat_layer(module):
            w, a = module.weight_quantizer, module.act_quantizer
            key = f"Kw={w.K}(m{w.m_neg},p{w.m_pos}),Ka={a.K}(m{a.m_neg},p{a.m_pos})"
            summary[key] = summary.get(key, 0) + 1
    return summary


def is_lcqat_state(state_dict: dict) -> bool:
    """True if a checkpoint state_dict was trained with LC-QAT.

    Matches on the *post* side only. A one-sided codebook (`m_neg=0`, which is
    exactly the MLP `relu^2` case) registers `raw_neg_deltas` as None and so has
    no such key, while a two-sided codebook always does. Using the post side
    keeps the detection correct for both shapes.
    """
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
    return PRESETS[DEFAULT_PRESET]


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
        if not is_lcqat_layer(module):
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
        # Sparse artifact (W3.4): the layer was exported as CSR over the
        # surviving index slots with a compacted alphabet. Carry the structure
        # through so the runtime can take the zero-skipping path; without these
        # the module would silently fall back to the dense interpretation of
        # `packed_weight_indices`.
        for suffix in (
            "sparse_keep_indices",
            "sparse_row_ptr",
            "sparse_col_indices",
            "sparse_index_format",
            "sparse_alphabet",
            "sparse_k_used",
        ):
            key = f"{name}.{suffix}"
            if key in model_data:
                module.register_buffer(suffix, model_data[key])
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

    The learned activation tables (D9) are attached on *both* retrofit paths
    here, not just in `finish_lcqat_after_load`. This function runs *before*
    `load_state_dict`, and the training checkpoint now carries
    `learnable_activation_lut.*` parameters -- so a resume that retrofitted
    without attaching them would fail the caller's `strict=True` load with
    "unexpected keys", and a resume that attached the wrong body would fail
    with a shape mismatch.
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
        _attach_luts(model, config)
        _patch_missing_lut_keys(model, model_data)
        return config
    return None


def _lut_key_groups(model: nn.Module) -> dict[str, list[str]]:
    """Expected activation-table keys, grouped by the sub-table that owns them.

    A flat key-level presence test is too blunt here, because the tables
    shipped in two generations: the baked `activation_lut` buffer first, then
    the learned `learnable_activation_lut.{logits,initial_table}` parameters. A
    checkpoint carrying the baked buffer and none of the learned parameters is
    a legitimate older format, not a corrupt one, and must still resume.
    Grouping by the sub-table prefix is what separates that case from genuine
    truncation, where only *part* of one table's keys survived.
    """
    groups: dict[str, list[str]] = {}
    for name in model.state_dict():
        if "activation_lut" not in name:
            continue
        if "learnable_activation_lut." in name:
            # `...c_fc.learnable_activation_lut.logits` -> the owning module, so
            # each layer's learned table is its own group. Splitting on a fixed
            # number of dots instead would merge layer 0 and layer 1 into one
            # group and make an ordinary two-layer checkpoint look truncated
            # whenever only one layer's table was absent.
            group = name.rsplit(".learnable_activation_lut.", 1)[0]
        else:
            group = name
        groups.setdefault(group, []).append(name)
    return groups


def _patch_missing_lut_keys(model: nn.Module, model_data: dict) -> int:
    """Fill D9 activation-table entries the checkpoint predates.

    `_attach_luts` installs the tables before the caller's `strict=True` load,
    so a checkpoint written before a given table existed is short exactly that
    table's keys and would otherwise fail with "missing keys" -- a resume that
    cannot resume. The defaults are the freshly attached tables themselves:
    `LearnableIndexLut` initializes bit-identical to `compile_activation_lut`,
    so a checkpoint with no table resumes at the behaviour it had before the
    table was learned, rather than at a different one.

    Only *missing* keys are filled. A key present in the checkpoint always
    wins, so this can never overwrite a trained table.

    An entirely absent sub-table is an older format and is patched. A
    *partially* present one is truncation, and filling the gap would blend
    half a trained table into half an untrained one, so it raises. Neither case
    is a config mismatch: a resume that rebuilt the wrong relaxation or body
    produces a model whose table keys the checkpoint does not have *and*
    checkpoint keys the model does not have, and that is left to the caller's
    `strict=True` load to report, so the error names the real problem.
    """
    groups = _lut_key_groups(model)
    expected = {n for names in groups.values() for n in names}
    # Keys the checkpoint has that this rebuild would not produce. Their
    # presence means the resume rebuilt the wrong relaxation or body, which is
    # a *config mismatch*, not truncation, and the caller's `strict=True` load
    # reports that accurately as an unexpected key. Patching anything while
    # such keys are present would paper over the real error with the wrong one.
    if any("activation_lut" in name and name not in expected for name in model_data):
        return 0
    missing: list[str] = []
    for group, names in groups.items():
        absent = [n for n in names if n not in model_data]
        if not absent:
            continue
        if len(absent) != len(names):
            raise RuntimeError(
                f"checkpoint carries {len(names) - len(absent)} of {len(names)} "
                f"keys for activation table {group!r}; {absent[:3]} are missing. "
                "A table is either absent entirely -- an older checkpoint format, "
                "which is patched -- or complete. A partial set is a truncated "
                "or corrupt checkpoint, and filling the gap would blend trained "
                "and untrained values. Retrain or re-save it."
            )
        missing.extend(absent)
    if not missing:
        return 0
    state = model.state_dict()
    with torch.no_grad():
        for name in missing:
            model_data[name] = state[name].detach().clone()
    # `warnings.warn` rather than the shell's `print0`: this is model-layer code,
    # and importing the logging shell here would invert the dependency direction.
    warnings.warn(
        f"Patched {len(missing)} missing activation-table entries to their "
        "initial values; this checkpoint predates those tables.",
        stacklevel=2,
    )
    return len(missing)


def _attach_luts(model: nn.Module, config: LayerKConfig) -> int:
    """Attach the D9 learned activation tables for `config`.

    Imported lazily: `export.py` imports from this module, so a module-level
    import here would be circular.
    """
    from nanochat.models.quant.export import attach_learnable_activation_luts

    return attach_learnable_activation_luts(
        model,
        relaxation=config.lut_relaxation,
        act_body=config.act_body,
    )


def finish_lcqat_after_load(
    model: nn.Module, requested: LayerKConfig | None
) -> LayerKConfig | None:
    """Retrofit a freshly loaded float checkpoint when QAT start was requested.

    Also attaches the D9 learned activation tables, because this is the resume
    path and it has the same obligation as the fresh-retrofit path: a resume
    that builds the layers but not their tables would resume with fewer
    parameters than it saved, and `verify_partition` would then flag the
    mismatch. Attached after `retrofit_model` for the same ordering reason.
    """
    if requested is None:
        return None
    retrofit_model(model, requested)
    _attach_luts(model, requested)
    return requested
