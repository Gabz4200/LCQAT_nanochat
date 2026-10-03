"""
Tests for per-layer retrofitting, optimizer grouping, and checkpoint helpers
(LC-QAT PRD sections 4 and 5).

python -m pytest tests/test_lcqat_retrofit.py -v
"""

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant import (
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
from nanochat.models.quant.optimizer import build_qat_param_groups, verify_partition
from nanochat.models.quant.retrofit import (
    DEFAULT_PRESET,
    get_layer_config,
    spec_k,
    spec_split,
)
from tests.conftest import (
    build_tiny_gpt,  # NOTE: structural only; gradient tests must use build_active_tiny_gpt (zero c_proj passes vacuously)
)


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
    # Asymmetric split syntax: m_neg-m_pos. The MLP `relu^2` tensors are the
    # case this exists for, so 0-7 (one-sided, 8 levels) must parse.
    assert parse_k_map("mlp.c_proj:0-7/0-7") == (("mlp.c_proj", (0, 7), (0, 7)),)
    assert parse_k_map("attn.c_q:6-8/6-8") == (("attn.c_q", (6, 8), (6, 8)),)
    # Even K is legal; 4 is a real 2-bit boundary, not an error.
    assert parse_k_map("mlp.c_proj:4/15") == (("mlp.c_proj", 4, 15),)
    with pytest.raises(ValueError, match=">= 3"):
        parse_k_map("mlp.c_proj:2/15")
    # A one-sided codebook needs two levels on its side, or the zero anchor
    # would also be the endpoint and bucketize would have no interior midpoint.
    with pytest.raises(ValueError, match="m_pos >= 2"):
        parse_k_map("mlp.c_proj:0-1/15")
    with pytest.raises(ValueError, match="substring:KW/KA"):
        parse_k_map("mlp.c_proj")


def test_when_asym_preset_then_non_negative_tensors_get_one_sided_codebooks() -> None:
    """`gpt.py` computes `relu(x).square()` before `mlp.c_proj`.

    The 4*n_embd hidden tensor is therefore non-negative, so both codebooks
    touching it (c_fc's output quantizer and c_proj's activation quantizer) get
    m_neg=0. A symmetric 15 spends 7 of its levels on a sign the tensor never
    takes; 0/7 spends all 8 on the range that exists.
    """
    config = PRESETS[DEFAULT_PRESET]
    c_fc = get_layer_config("transformer.h.0.mlp.c_fc", config)
    assert c_fc is not None
    assert c_fc.quantize_output is True
    assert c_fc.output == (0, 7), "c_fc output sees relu^2 >= 0"

    c_proj = get_layer_config("transformer.h.0.mlp.c_proj", config)
    assert c_proj is not None
    assert c_proj.activation == (0, 7), "c_proj input sees relu^2 >= 0"

    # Attention tensors are genuinely signed, so they keep both sides.
    for name in ("transformer.h.0.attn.c_q", "transformer.h.0.attn.c_proj"):
        spec = get_layer_config(name, config)
        assert spec is not None
        assert spec.activation[0] > 0, f"{name} input is RMSNorm'd and signed"


def test_when_asym_preset_then_applied_to_a_model() -> None:
    model = retrofit_model(build_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    c_fc = model.get_submodule("transformer.h.0.mlp.c_fc")
    c_proj = model.get_submodule("transformer.h.0.mlp.c_proj")
    assert isinstance(c_fc, LCQATLinear) and isinstance(c_proj, LCQATLinear)
    assert c_fc.out_quantizer.m_neg == 0
    assert c_fc.out_quantizer.m_pos == 7
    assert c_proj.act_quantizer.m_neg == 0
    assert c_proj.act_quantizer.m_pos == 7
    # The one-sided codebook's levels are all >= 0, which is the point: a
    # symmetric 15 would place 7 levels below the minimum the tensor can reach.
    assert (c_proj.act_quantizer.get_codebook() >= 0).all()
    assert c_proj.act_quantizer.get_codebook()[0].item() == 0.0
    # The forward agrees: a non-negative activation never produces a negative
    # dequantized value.
    with torch.no_grad():
        pos = torch.rand(2, 16, c_proj.in_features)
        assert (c_proj.act_quantizer(pos).value >= 0).all()


def test_when_spec_helpers_then_k_and_split_are_consistent() -> None:
    assert spec_k(15) == 15
    assert spec_k((0, 7)) == 8
    assert spec_split(15) == (7, 7)
    # Even K cannot be symmetric; the extra level goes positive.
    assert spec_split(8) == (3, 4)
    assert sum(spec_split(8)) + 1 == spec_k(8)
    assert spec_split((6, 8)) == (6, 8)
    with pytest.raises(ValueError, match=">= 3"):
        spec_split(2)


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


def test_when_build_qat_param_groups_then_codebooks_get_their_own_adamw_group(
    tiny_gpt, tiny_gpt_lcqat
) -> None:
    """Codebook params get a dedicated group (PRD 5: own LR, no weight decay).

    Runs against `build_qat_param_groups`, which is the real integration point.
    `GPT.setup_optimizer` was deleted: it only ever saw `transformer.h`, so it
    would silently drop the DiffusionBlocks adapters and denoise heads.
    """
    groups = build_qat_param_groups(tiny_gpt, matrix_lr=0.02, weight_decay=0.1)
    codebook_params = {id(p) for n, p in tiny_gpt.named_parameters() if "deltas" in n}
    assert not codebook_params
    assert not any(
        any(id(p) in codebook_params for p in group["params"]) for group in groups
    )
    verify_partition(tiny_gpt, groups)

    groups = build_qat_param_groups(
        tiny_gpt_lcqat, matrix_lr=0.02, weight_decay=0.1, codebook_lr=1e-3
    )
    codebook_params = {
        id(p) for n, p in tiny_gpt_lcqat.named_parameters() if "deltas" in n
    }
    groups_with_codebooks = [
        group
        for group in groups
        if any(id(p) in codebook_params for p in group["params"])
    ]
    assert len(groups_with_codebooks) == 1
    group = groups_with_codebooks[0]
    assert group["kind"] == "adamw"
    assert group["role"] == "codebook"
    assert group["weight_decay"] == 0.0
    assert group["lr"] > 0
    assert {id(p) for p in group["params"]} == codebook_params
    # Every parameter lands in exactly one group (this is the check that caught
    # the duplicate-codebook-parameter bug).
    verify_partition(tiny_gpt_lcqat, groups)
    covered = {id(p) for group in groups for p in group["params"]}
    assert covered == {id(p) for p in tiny_gpt_lcqat.parameters()}


def test_when_build_qat_param_groups_then_per_role_lrs_are_not_dropped(
    tiny_gpt,
) -> None:
    """Regression for the silently-discarded `--embedding-lr` family.

    The previous two-group builder ignored embedding_lr / unembedding_lr /
    scalar_lr entirely, so those flags were parsed and thrown away.
    """
    groups = build_qat_param_groups(
        tiny_gpt,
        matrix_lr=0.02,
        weight_decay=0.1,
        embedding_lr=0.3,
        unembedding_lr=0.008,
        scalar_lr=0.5,
    )
    lrs = {g["role"]: g["lr"] for g in groups}
    assert lrs["embed"] == pytest.approx(0.3)
    assert lrs["lm_head"] == pytest.approx(0.008)
    assert lrs["x0"] == pytest.approx(0.5)
    assert lrs["resid"] == pytest.approx(0.5 * 0.01)
    assert lrs["matrix"] == pytest.approx(0.02)
    # dmodel_lr_scale multiplies the AdamW roles but not the scalar ones,
    # matching GPT.setup_optimizer's tuned recipe.
    scaled = build_qat_param_groups(
        tiny_gpt, matrix_lr=0.02, weight_decay=0.1, dmodel_lr_scale=2.0
    )
    s_lrs = {g["role"]: g["lr"] for g in scaled}
    assert s_lrs["embed"] == pytest.approx(0.6)
    assert s_lrs["x0"] == pytest.approx(0.5)


@pytest.mark.slow
def test_when_optimizer_step_then_codebook_params_change(tiny_gpt_lcqat) -> None:
    torch.manual_seed(0)
    model = tiny_gpt_lcqat
    optimizer = torch.optim.AdamW(
        build_qat_param_groups(model, matrix_lr=0.02, weight_decay=0.1)
    )

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
    # No meta and no requested config: the active config falls back to the
    # default preset, which is now `asym` (not `small`). The roundtrip below
    # therefore only holds because the fixture was built with that same preset.
    active = prepare_lcqat_before_load(fresh, state, None, None)
    assert active == PRESETS[DEFAULT_PRESET]
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
    """The mutual-exclusion guard, against the class `--fp8` actually installs.

    Previously this test used a stand-in that only borrowed the name, so it
    pinned the name heuristic rather than the contract. Both paths are covered
    below: the real class, and a look-alike that only carries the name.
    """
    from nanochat.models.fp8 import Float8Linear

    tiny_gpt.lm_head = Float8Linear(256, 128, bias=False)
    # route it through a role that would otherwise be converted
    tiny_gpt.transformer.h[0].mlp.c_fc = Float8Linear(256, 1024, bias=False)
    with pytest.raises(ValueError, match="fp8"):
        retrofit_model(tiny_gpt, PRESETS["small"])


def test_when_a_lookalike_float8_class_then_retrofit_also_raises(tiny_gpt) -> None:
    """A class from a build that does not expose `Float8Linear` for import.

    `--fp8` layers are identified by name elsewhere in the codebase
    (`scripts/_train/build.py`'s `num_fp8` count, `disable_fp8`), so the guard
    keeps that as a fallback rather than relying on the import alone.
    """

    class SomeOtherBuildFloat8Linear(nn.Linear):
        pass

    tiny_gpt.transformer.h[0].mlp.c_fc = SomeOtherBuildFloat8Linear(
        256, 1024, bias=False
    )
    with pytest.raises(ValueError, match="fp8"):
        retrofit_model(tiny_gpt, PRESETS["small"])


def test_when_a_plain_linear_then_the_fp8_guard_does_not_fire(tiny_gpt) -> None:
    """A class whose name merely resembles fp8 must not be rejected."""
    assert retrofit_model(tiny_gpt, PRESETS["small"]) is tiny_gpt
