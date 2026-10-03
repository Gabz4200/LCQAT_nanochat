"""End-to-end wiring of `--lcqat-channel-center` (handoff §9.3, Phase 2).

The flag's three failure modes, each pinned here:

1. **Registered but not consumed.** `lcqat_config_from_args` never read it, so
   the flag looked implemented and built a shared table. A flag in that state is
   worse than an absent one (handoff §13), so `lcqat_config_from_args` now
   consumes it and these tests assert the config actually changes.
2. **Dropped on resume.** The per-channel parameters live in the state_dict; a
   reload that rebuilt shared tables would orphan them silently. So the config
   must survive the `as_dict` -> JSON -> `from_dict` round trip.
3. **Silently mis-exported.** `weight_quantizer` stays present on a per-channel
   layer (it keeps `K_weight` and the optimizer partition readable), so the
   exporter would pack indices from the *shared* table and emit an artifact
   whose indices mean something different from the trained weights. Export must
   raise instead.
"""

from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.export import export_lcqat_checkpoint
from nanochat.models.quant.linear import LCQATLinear
from nanochat.models.quant.per_channel import PerChannelValueCenteredQuantizer
from nanochat.models.quant.retrofit import (
    CODEBOOK_SPEC_FIELDS,
    DEFAULT_PRESET,
    PRESETS,
    LayerKConfig,
    _validate_codebook_spec,
    lcqat_config_from_args,
    retrofit_model,
)
from tests.conftest import build_active_tiny_gpt


def _args(**overrides) -> argparse.Namespace:
    """Parsed-flags stand-in. Only the keys `lcqat_config_from_args` reads."""
    base = {
        "lcqat_preset": DEFAULT_PRESET,
        "lcqat_k_map": "",
        "codebook_grad_scale": "inv_sqrt_n",
        "lcqat_channel_center": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _lcqat_linears(model: nn.Module) -> list[LCQATLinear]:
    return [m for m in model.modules() if isinstance(m, LCQATLinear)]


# --- 1. the flag is consumed ------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, False, id="flag_absent"),
        pytest.param({"lcqat_channel_center": True}, True, id="flag_set"),
        pytest.param({"lcqat_channel_center": False}, False, id="flag_off"),
        # `chat_rl` does not register `--lcqat-channel-center` at all.
        # `lcqat_config_from_args` reads it with a `getattr` default precisely so
        # a caller that registered a subset of the flags still works; without
        # that, the missing attribute would be an AttributeError at startup.
        pytest.param({"__delete__": True}, False, id="flag_not_registered"),
    ],
)
def test_the_channel_center_flag_decides_per_channel_tables(
    kwargs: dict, expected: bool
) -> None:
    args = _args(**{k: v for k, v in kwargs.items() if k != "__delete__"})
    if "__delete__" in kwargs:
        del args.lcqat_channel_center
    assert lcqat_config_from_args(args).per_channel_weight is expected


def test_when_flag_set_then_retrofitted_model_really_gets_per_channel_tables() -> None:
    cfg = lcqat_config_from_args(_args(lcqat_channel_center=True))
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    layers = _lcqat_linears(model)
    assert layers, "fixture produced no LCQATLinear modules"

    per_channel = [
        m
        for m in layers
        if isinstance(m.per_channel_weight_quantizer, PerChannelValueCenteredQuantizer)
    ]
    assert len(per_channel) == len(layers), (
        "every retrofitted layer must carry a per-channel table when the flag "
        "is on; a partial retrofit would train two different quantizers under "
        "one config"
    )
    for module in per_channel:
        assert module.per_channel_weight_quantizer.num_channels == module.out_features
        # The shared table is retained on purpose (K_weight, the optimizer
        # partition and the export planner all still read it), so its presence
        # is not evidence of anything -- the per-channel table's own shape is.
        assert module.weight_quantizer.K == module.K_weight


def test_when_flag_off_then_no_layer_gets_a_per_channel_table() -> None:
    model = retrofit_model(build_active_tiny_gpt(), lcqat_config_from_args(_args()))
    for module in _lcqat_linears(model):
        assert module.per_channel_weight_quantizer is None


# --- 2. it survives the checkpoint round trip -------------------------------


def test_when_config_is_serialized_then_per_channel_survives_json() -> None:
    """A resume must rebuild per-channel tables, not shared ones.

    `resolve_lcqat_config` prefers the checkpoint's `meta["lcqat"]`, so a flag
    that did not survive `as_dict`/`from_dict` would be dropped on every resume
    and the saved per-channel parameters would be orphaned -- silently, since
    a shared table still trains and still saves.
    """
    original = lcqat_config_from_args(_args(lcqat_channel_center=True))
    # Round-trip through plain JSON types, exactly as a checkpoint writes them.
    revived = LayerKConfig.from_dict(original.as_dict())
    assert revived.per_channel_weight is True
    assert revived == original


def test_when_per_channel_config_is_loaded_then_retrofit_builds_per_channel() -> None:
    cfg = LayerKConfig.from_dict(
        lcqat_config_from_args(_args(lcqat_channel_center=True)).as_dict()
    )
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    assert all(
        m.per_channel_weight_quantizer is not None for m in _lcqat_linears(model)
    )


def test_when_per_channel_layer_forwards_then_uses_the_per_channel_table() -> None:
    """The forward must route through the per-channel table, not the shared one.

    A per-channel quantizer that was built but not used would make the flag
    inert again -- exactly the failure mode this wiring exists to close. Proved
    by perturbation rather than by shape: the output is recomputed after a
    known change to the *per-channel* table's own parameter, and the change
    must reach the output. Poking the shared `weight_quantizer` instead would
    leave the output identical, which is precisely the bug this rules out.
    """
    cfg = lcqat_config_from_args(_args(lcqat_channel_center=True))
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    module = _lcqat_linears(model)[0]
    module.per_channel_weight_quantizer.init_from_tensor(module.weight)
    module.per_channel_weight_quantizer.train()

    x = torch.randn(2, 3, module.in_features)
    before = module(x).detach().clone()

    with torch.no_grad():
        target = module.per_channel_weight_quantizer.raw_pos
        assert target is not None and target.numel() > 0, (
            "expected trainable positive-arm deltas on the per-channel table"
        )
        target.add_(0.5)
    after = module(x).detach()

    assert before.shape == after.shape == (2, 3, module.out_features)
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert not torch.allclose(before, after), (
        "perturbing the per-channel weight table did not change the output, so "
        "forward is not reading it -- the flag is decorative"
    )
    out = module(x)
    assert out.shape == (2, 3, module.out_features)
    assert torch.isfinite(out).all()


def test_when_per_channel_layer_backprops_then_per_channel_table_gets_grads() -> None:
    cfg = lcqat_config_from_args(_args(lcqat_channel_center=True))
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    module = _lcqat_linears(model)[0]
    # Per-channel tables are biased by observed statistics; without this every
    # value buckets to the anchor, the anchor is the only gathered level, and
    # the table receives exactly zero gradient (handoff §9.8: silent and
    # permanent).
    module.per_channel_weight_quantizer.init_from_tensor(module.weight)
    module.per_channel_weight_quantizer.train()

    x = torch.randn(2, 3, module.in_features)
    module(x).square().mean().backward()

    grads = [
        p.grad
        for p in module.per_channel_weight_quantizer.parameters()
        if p.requires_grad
    ]
    assert grads, "per-channel table exposes no trainable parameters"
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        "no per-channel parameter received gradient; the table is inert and "
        "the flag is decorative"
    )


# --- 3. it refuses to mis-export --------------------------------------------


def test_when_exporting_a_per_channel_model_then_raises(tmp_path) -> None:
    cfg = lcqat_config_from_args(_args(lcqat_channel_center=True))
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    with pytest.raises(ValueError, match="per-channel"):
        export_lcqat_checkpoint(model, str(tmp_path / "artifact.pt"))


def test_when_per_channel_meets_packed_indices_then_forward_raises() -> None:
    """The inference-side guard: a [C, K] table cannot be packed as [K] ids."""
    module = LCQATLinear(256, 256, K_weight=15, K_act=15, per_channel_weight=True)
    module.register_buffer(
        "packed_weight_indices",
        torch.zeros(module.out_features, module.in_features, dtype=torch.uint8),
        persistent=True,
    )
    module.register_buffer("weight_index_format", torch.tensor(0, dtype=torch.int64))
    with pytest.raises(ValueError, match="per-channel"):
        module(torch.randn(2, 4, module.in_features))


# --- 4. the landmine itself -------------------------------------------------


def test_when_validating_then_bool_field_named_like_a_codebook_is_not_one() -> None:
    """`per_channel_weight` ends in `_weight`; the suffix heuristic ate it.

    Handoff §9.3 Defect 1: with a suffix-matching `validate()`, plain
    `--lcqat-preset asym` raised `ValueError: ...per_channel_weight must be an
    int K ... got False` and `base_train` could not start at all. The explicit
    `CODEBOOK_SPEC_FIELDS` set makes that class of bug unrepresentable, so this
    pins the fix at the exact call that used to throw.
    """
    for name in PRESETS:
        PRESETS[name].validate()  # the reproducer from the handoff


def test_when_codebook_spec_fields_are_declared_then_all_are_real_fields() -> None:
    """Every spec field is validated; the non-spec fields are not.

    Stated positively rather than reverse-engineered from field types: a type
    oracle cannot decide membership here. `k_map` defaults to `()` (a tuple,
    so "is a tuple" would admit it) and `min_linear_dim` to `128` (an int, so
    "is an int" would admit it), while `per_channel_weight` is `False` (a bool
    the suffix heuristic wrongly admitted). The set is the contract; this test
    pins both directions of it.
    """
    fields_map = LayerKConfig.__dataclass_fields__
    assert CODEBOOK_SPEC_FIELDS <= set(fields_map)

    # Validated as CodebookSpecs: exactly the 12 role fields.
    assert set(CODEBOOK_SPEC_FIELDS) == {
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
    # Deliberately excluded: `per_channel_weight` (the §9.3 landmine),
    # `min_linear_dim` and the two `quantize_*` booleans (not specs), and
    # `k_map` (a tuple of rules, validated entry-by-entry just above).
    assert set(CODEBOOK_SPEC_FIELDS).isdisjoint(
        {
            "per_channel_weight",
            "min_linear_dim",
            "quantize_qkv_out",
            "quantize_fc_out",
            "k_map",
        }
    )


def test_when_a_bool_is_passed_as_a_codebook_spec_then_it_raises_clearly() -> None:
    """Defense in depth: `bool` is an `int` subclass, so `False` is a "K of 0"."""
    with pytest.raises(ValueError, match="is a bool"):
        _validate_codebook_spec(False, "cfg.x_weight")


def test_when_per_channel_is_set_on_a_config_directly_then_validate_passes() -> None:
    LayerKConfig(per_channel_weight=True).validate()


def test_when_grad_scale_is_invalid_then_validate_still_raises() -> None:
    """The other half of `validate()` survived the rewrite."""
    with pytest.raises(ValueError, match="grad_scale"):
        LayerKConfig(grad_scale="bogus").validate()
