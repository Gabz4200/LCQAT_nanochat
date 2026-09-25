"""
Tests for per-layer retrofitting, optimizer grouping, and checkpoint helpers
(LC-QAT PRD sections 4 and 5).

python -m pytest tests/test_lcqat_retrofit.py -v
"""

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from nanochat.lcqat import (
    PRESETS,
    LCQATLinear,
    finish_lcqat_after_load,
    is_lcqat_state,
    lcqat_config_from_args,
    parse_k_map,
    prepare_lcqat_before_load,
    retrofit_model,
    retrofit_summary,
)
from nanochat.lcqat.retrofit import get_layer_config


def _k_of(model: nn.Module, name: str) -> tuple[int, int]:
    module = model.get_submodule(name)
    assert isinstance(module, LCQATLinear), f"{name} is {type(module).__name__}"
    return module.K_weight, module.K_act


def test_when_small_preset_then_role_mapping_matches(tiny_gpt) -> None:
    retrofit_model(tiny_gpt, PRESETS["small"])
    assert _k_of(tiny_gpt, "transformer.h.0.attn.c_q") == (3, 15)
    assert _k_of(tiny_gpt, "transformer.h.1.attn.c_k") == (3, 15)
    assert _k_of(tiny_gpt, "transformer.h.0.attn.c_v") == (15, 15)
    assert _k_of(tiny_gpt, "transformer.h.0.attn.c_proj") == (15, 15)
    assert _k_of(tiny_gpt, "transformer.h.0.mlp.c_fc") == (15, 15)
    assert _k_of(tiny_gpt, "transformer.h.0.mlp.c_proj") == (15, 15)


def test_when_prd_preset_then_down_projection_gets_8bit_codebooks(tiny_gpt) -> None:
    retrofit_model(tiny_gpt, PRESETS["prd"])
    assert _k_of(tiny_gpt, "transformer.h.0.mlp.c_proj") == (255, 255)
    assert _k_of(tiny_gpt, "transformer.h.0.attn.c_q") == (3, 15)


def test_when_retrofitting_then_lm_head_and_tiny_linears_stay_float(tiny_gpt) -> None:
    retrofit_model(tiny_gpt, PRESETS["small"])
    assert not isinstance(tiny_gpt.lm_head, LCQATLinear)
    assert not isinstance(tiny_gpt.smear_gate, LCQATLinear)
    block0 = tiny_gpt.transformer.h[0]
    if block0.attn.ve_gate is not None:
        assert not isinstance(block0.attn.ve_gate, LCQATLinear)


def test_when_qkv_and_fc_then_output_quantizers_exist(tiny_gpt) -> None:
    retrofit_model(tiny_gpt, PRESETS["small"])
    block = tiny_gpt.transformer.h[0]
    assert block.attn.c_q.out_quantizer is not None  # training quantizes Q/K/V outputs
    assert block.attn.c_k.out_quantizer is not None
    assert block.attn.c_v.out_quantizer is not None
    assert (
        block.mlp.c_fc.out_quantizer is not None
    )  # input side of the fused activation LUT
    assert block.attn.c_proj.out_quantizer is None
    assert block.mlp.c_proj.out_quantizer is None


def test_when_retrofit_twice_then_is_idempotent(tiny_gpt) -> None:
    retrofit_model(tiny_gpt, PRESETS["small"])
    summary = retrofit_summary(tiny_gpt)
    retrofit_model(tiny_gpt, PRESETS["small"])
    assert retrofit_summary(tiny_gpt) == summary


def test_when_k_map_rule_matches_nothing_then_raises(tiny_gpt) -> None:
    config = replace(PRESETS["small"], k_map=(("does.not.exist", 3, 15),))
    with pytest.raises(ValueError, match="matched no modules"):
        retrofit_model(tiny_gpt, config)


def test_when_k_map_overrides_role_defaults(tiny_gpt) -> None:
    config = replace(PRESETS["small"], k_map=(("mlp.c_proj", 255, 255),))
    retrofit_model(tiny_gpt, config)
    assert _k_of(tiny_gpt, "transformer.h.0.mlp.c_proj") == (255, 255)
    assert _k_of(tiny_gpt, "transformer.h.0.attn.c_q") == (3, 15)


def test_when_get_layer_config_for_unmatched_names_then_skips() -> None:
    cfg = PRESETS["small"]
    assert get_layer_config("lm_head", cfg) is None
    assert get_layer_config("smear_gate", cfg) is None
    assert get_layer_config("transformer.h.0.attn.ve_gate", cfg) is None


def test_when_parsing_k_map_then_validates_entries() -> None:
    assert parse_k_map("mlp.c_proj:255/255, attn.c_q:3/15") == (
        ("mlp.c_proj", 255, 255),
        ("attn.c_q", 3, 15),
    )
    with pytest.raises(ValueError, match="odd"):
        parse_k_map("mlp.c_proj:4/15")
    with pytest.raises(ValueError, match="substring:KW/KA"):
        parse_k_map("mlp.c_proj")


def test_when_config_from_args_then_preset_and_k_map_apply() -> None:
    args = type("Args", (), {"lcqat_preset": "prd", "lcqat_k_map": "attn.c_q:15/15"})()
    config = lcqat_config_from_args(args)
    assert config.down_weight == 255
    assert config.k_map == (("attn.c_q", 15, 15),)
    with pytest.raises(ValueError, match="Unknown --lcqat-preset"):
        lcqat_config_from_args(
            type("Args", (), {"lcqat_preset": "bogus", "lcqat_k_map": ""})()
        )


def test_when_forward_and_backward_on_retrofitted_model_then_codebooks_get_grads(
    tiny_gpt_lcqat,
) -> None:
    model = tiny_gpt_lcqat
    # nanochat zero-initializes the attention/MLP projections, and a zero weight
    # matrix transmits exactly zero gradient upstream, so on the very first step
    # nothing behind the projections (including codebooks) receives any gradient.
    # Nudge them off zero the way the first optimizer step would.
    torch.manual_seed(0)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.add_(
                0.01 * torch.randn_like(block.attn.c_proj.weight)
            )
            block.mlp.c_proj.weight.add_(
                0.01 * torch.randn_like(block.mlp.c_proj.weight)
            )
    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    codebook_grads = [
        p.grad.abs().sum()
        for n, p in model.named_parameters()
        if "deltas" in n and p.grad is not None
    ]
    assert len(codebook_grads) > 0
    assert sum(g.item() for g in codebook_grads) > 0
    weight = model.transformer.h[0].mlp.c_fc.weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_when_num_scaling_params_then_codebooks_are_tracked_separately(
    tiny_gpt, tiny_gpt_lcqat
) -> None:
    float_counts = tiny_gpt.num_scaling_params()
    lcqat_counts = tiny_gpt_lcqat.num_scaling_params()
    assert lcqat_counts["codebooks"] > 0
    assert float_counts["codebooks"] == 0
    assert lcqat_counts["transformer_matrices"] == float_counts["transformer_matrices"]
    assert lcqat_counts["total"] == sum(p.numel() for p in tiny_gpt_lcqat.parameters())


def test_when_setup_optimizer_then_codebooks_get_their_own_adamw_group(
    tiny_gpt, tiny_gpt_lcqat
) -> None:
    optimizer = tiny_gpt.setup_optimizer()
    codebook_params = {id(p) for n, p in tiny_gpt.named_parameters() if "deltas" in n}
    assert not any(
        any(id(p) in codebook_params for p in group["params"])
        for group in optimizer.param_groups
    )

    optimizer = tiny_gpt_lcqat.setup_optimizer()
    codebook_params = {
        id(p) for n, p in tiny_gpt_lcqat.named_parameters() if "deltas" in n
    }
    groups_with_codebooks = [
        group
        for group in optimizer.param_groups
        if any(id(p) in codebook_params for p in group["params"])
    ]
    assert len(groups_with_codebooks) == 1
    group = groups_with_codebooks[0]
    assert group["kind"] == "adamw"
    assert group["weight_decay"] == 0.0
    assert group["lr"] > 0
    assert {id(p) for p in group["params"]} == codebook_params
    # Muon must only ever see 2-D matrices
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            assert all(p.ndim == 2 for p in group["params"])
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert covered == {id(p) for p in tiny_gpt_lcqat.parameters()}


@pytest.mark.slow
def test_when_optimizer_step_then_codebook_params_change(tiny_gpt_lcqat) -> None:
    torch.manual_seed(0)
    model = tiny_gpt_lcqat
    optimizer = model.setup_optimizer()

    def step() -> None:
        idx = torch.randint(0, model.config.vocab_size, (2, 16))
        targets = torch.randint(0, model.config.vocab_size, (2, 16))
        loss = model(idx, targets)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

    # Step 1 only moves the zero-initialized projections off zero (zero weights
    # transmit zero gradient upstream, so codebook grads are still zero there).
    step()
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if "deltas" in n
    }
    # Step 2: gradients now reach the codebooks and the optimizer updates them.
    step()
    changed = any(
        not torch.equal(before[n], p)
        for n, p in model.named_parameters()
        if n in before
    )
    assert changed, "no codebook parameter moved after optimizer.step()"


def test_when_checkpoint_helpers_then_state_detection_and_roundtrip_work(
    tiny_gpt, tiny_gpt_lcqat, tiny_gpt_factory
) -> None:
    state = {k: v.clone() for k, v in tiny_gpt_lcqat.state_dict().items()}
    assert is_lcqat_state(state)
    assert not is_lcqat_state(tiny_gpt.state_dict())

    fresh = tiny_gpt_factory()
    active = prepare_lcqat_before_load(fresh, state, None, None)
    assert active == PRESETS["small"]
    fresh.load_state_dict(state, strict=True)
    fresh.eval()
    tiny_gpt_lcqat.eval()
    idx = torch.randint(0, fresh.config.vocab_size, (2, 16))
    with torch.no_grad():
        assert torch.equal(fresh(idx), tiny_gpt_lcqat(idx))

    plain = tiny_gpt_factory()
    assert prepare_lcqat_before_load(plain, plain.state_dict(), None, None) is None
    assert finish_lcqat_after_load(plain, None) is None
    active = finish_lcqat_after_load(plain, PRESETS["prd"])
    assert active == PRESETS["prd"]
    assert _k_of(plain, "transformer.h.0.mlp.c_proj") == (255, 255)


def test_when_meta_config_present_then_it_wins_over_requested(
    tiny_gpt_factory,
) -> None:
    from dataclasses import asdict

    # checkpoint trained with the prd preset; flags ask for small -> meta wins
    donor = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    state = donor.state_dict()
    fresh = tiny_gpt_factory()
    active = prepare_lcqat_before_load(
        fresh, state, asdict(PRESETS["prd"]), PRESETS["small"]
    )
    assert active == PRESETS["prd"]
    fresh.load_state_dict(state, strict=True)
    assert _k_of(fresh, "transformer.h.0.mlp.c_proj") == (255, 255)


def test_when_float8_module_then_retrofit_raises(tiny_gpt) -> None:
    class FakeFloat8Linear(nn.Linear):
        pass

    tiny_gpt.lm_head = FakeFloat8Linear(256, 128, bias=False)
    # route it through a role that would otherwise be converted
    tiny_gpt.transformer.h[0].mlp.c_fc = FakeFloat8Linear(256, 1024, bias=False)
    with pytest.raises(ValueError, match="fp8"):
        retrofit_model(tiny_gpt, PRESETS["small"])
