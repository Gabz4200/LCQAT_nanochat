"""Tests for EfQAT selective layer freezing (PRD section 3.2)."""

import torch

from nanochat.models.backbone import GPT, GPTConfig
from nanochat.models.quant import PRESETS, retrofit_model
from nanochat.models.quant.efqat import CRITICAL_PATTERNS, SelectiveFreezer


def _tiny_lcqat_model():
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=6,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return retrofit_model(model, PRESETS["small"])


def _non_critical_codebook_params(layer):
    """Codebook deltas from non-critical projections (c_v, mlp) in a layer."""
    out = []
    for name, p in layer.named_parameters():
        if not name.endswith(("raw_pos_deltas", "raw_neg_deltas")):
            continue
        # skip critical outlier projections
        if any(pat in name for pat in ("c_q", "c_k", "wte", "lm_head")):
            continue
        out.append(p)
    return out


def test_selective_freezer_freezes_middle_after_warmup():
    model = _tiny_lcqat_model()
    freezer = SelectiveFreezer(model, warmup_steps=5, freeze_middle_frac=0.5)
    n_layer, start, end = freezer.layer_bounds()
    assert n_layer == len(model.transformer.h)
    assert end > start
    # Before warmup: nothing frozen.
    assert freezer.update(3) is False
    assert not freezer.is_frozen()
    # At warmup: middle band freezes.
    assert freezer.update(5) is True
    assert freezer.is_frozen()
    middle = model.transformer.h[start]
    frozen_params = _non_critical_codebook_params(middle)
    assert len(frozen_params) > 0
    assert all(not p.requires_grad for p in frozen_params)


def test_selective_freezer_keeps_boundaries_critical():
    model = _tiny_lcqat_model()
    n_layer = len(model.transformer.h)
    freezer = SelectiveFreezer(model, warmup_steps=0, freeze_middle_frac=0.5)
    freezer.update(0)
    assert freezer.is_frozen()
    _, start, end = freezer.layer_bounds()
    # Layer 0 (input) and last layer (output) are outside the frozen band.
    for li in (0, n_layer - 1):
        # c_q / c_k are critical: codebook deltas must stay trainable.
        params = [
            p
            for n, p in model.transformer.h[li].named_parameters()
            if n.endswith(("raw_pos_deltas", "raw_neg_deltas")) and "c_q" in n
        ]
        assert params, f"layer {li} should have c_q codebook params"
        assert all(p.requires_grad for p in params), (
            f"boundary {li} c_q should stay trainable"
        )


def test_selective_freezer_noop_when_frac_zero():
    model = _tiny_lcqat_model()
    freezer = SelectiveFreezer(model, warmup_steps=0, freeze_middle_frac=0.0)
    freezer.update(0)
    assert freezer.is_frozen()
    # frac=0 -> empty band -> no params frozen
    assert len(freezer._frozen_params) == 0


def test_selective_freezer_unfreeze_reverses():
    model = _tiny_lcqat_model()
    freezer = SelectiveFreezer(model, warmup_steps=0, freeze_middle_frac=0.5)
    freezer.update(0)
    n_frozen = len(freezer._frozen_params)
    assert n_frozen > 0
    unfrozen = freezer.unfreeze()
    assert unfrozen == n_frozen
    assert not freezer.is_frozen()


def test_critical_patterns_include_spec():
    for pat in ("wte", "c_q", "c_k", "lm_head"):
        assert any(pat == c or pat in c for c in CRITICAL_PATTERNS)


def test_selective_freezer_warmup_zero_freezes_immediately():
    model = _tiny_lcqat_model()
    freezer = SelectiveFreezer(model, warmup_steps=0, freeze_middle_frac=0.5)
    assert freezer.update(0) is True
    assert freezer.is_frozen()
