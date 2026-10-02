"""The optimizer role partition (W2.6).

`build_qat_param_groups` replaced `GPT.setup_optimizer`, which emitted exactly two
groups (matrix, codebook). Consequences of that, both regressions here:

* `--embedding-lr` / `--unembedding-lr` / `--scalar-lr` were parsed by
  `base_train` and then silently discarded, so those parameters trained at the
  matrix LR.
* `GPT.setup_optimizer` only ever saw `transformer.h`, so the DiffusionBlocks
  adapters and per-block denoise heads fell outside every group and never
  received an update.

`verify_partition` is the guard: it asserts every parameter lands in exactly one
group. That check is what caught the duplicate-codebook-parameter bug when the
engine's LC-QAT layers were substituted rather than re-parented in place.
"""

import pytest
import torch

from nanochat.diffusion_blocks import DiffusionBlockEngine, EquiProbabilityPartitioner
from nanochat.lcqat import PRESETS, retrofit_model
from nanochat.lcqat.optimizer import (
    ROLE_ORDER,
    build_qat_param_groups,
    is_codebook_param,
    role_for_name,
    verify_partition,
)
from nanochat.lcqat.retrofit import DEFAULT_PRESET
from tests.conftest import build_active_tiny_gpt


def _engine(num_blocks=2, sparsity=0.75):
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=num_blocks)
    )
    engine.apply_lcqat(PRESETS[DEFAULT_PRESET])
    if sparsity:
        engine.apply_sparseprop(sparsity=sparsity, with_lcqat=True)
    return engine


def test_when_role_for_name_then_each_name_maps_to_one_role():
    assert role_for_name("transformer.h.0.mlp.c_fc.weight") == "matrix"
    assert (
        role_for_name("transformer.h.0.mlp.c_fc.weight_quantizer.raw_pos_deltas")
        == "codebook"
    )
    assert role_for_name("transformer.wte.weight") == "embed"
    assert role_for_name("lm_head.weight") == "lm_head"
    assert role_for_name("value_embeds.1.weight") == "value_embed"
    assert role_for_name("resid_lambdas") == "resid"
    assert role_for_name("x0_lambdas") == "x0"
    assert role_for_name("smear_lambda") == "smear"
    assert role_for_name("backout_lambda") == "smear"
    # Engine-owned params train like matrices but are not transformer layers.
    assert role_for_name("db_adapters.0.mlp.0.weight") == "matrix"
    assert role_for_name("db_denoise_heads.1.weight") == "matrix"
    for name in ("db_adapters.0.mlp.0.weight", "db_denoise_heads.1.weight"):
        assert role_for_name(name) in ROLE_ORDER


def test_when_is_codebook_param_then_both_sides_are_detected():
    assert is_codebook_param("a.raw_pos_deltas")
    assert is_codebook_param("a.raw_neg_deltas")
    assert not is_codebook_param("a.weight")


def test_when_engine_with_all_three_technologies_then_partition_is_complete():
    """LC-QAT + SparseProp + DiffusionBlocks, the actual combined engine."""
    engine = _engine()
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    verify_partition(engine, groups)
    seen = {id(p) for g in groups for p in g["params"]}
    assert seen == {id(p) for p in engine.parameters()}


def test_when_engine_then_no_parameter_is_duplicated_across_groups():
    """Regression for the in-place sparse wrapping.

    Substituting a `SparsePropLinearLCQAT` for an `LCQATLinear` already installed
    in a live `nn.Sequential` slot left the original registered at its old path,
    making every codebook parameter reachable twice. AdamW rejects that, but
    only at construction; this catches it earlier and more clearly.
    """
    engine = _engine()
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    seen = {}
    for g in groups:
        for p in g["params"]:
            pid = id(p)
            assert pid not in seen, f"{pid} in both {seen[pid]} and {g['role']}"
            seen[pid] = g["role"]


def test_when_verify_partition_sees_a_missing_group_then_raises():
    engine = _engine()
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    stripped = [g for g in groups if g["role"] != "embed"]
    with pytest.raises(AssertionError, match="no optimizer group"):
        verify_partition(engine, stripped)


def test_when_verify_partition_sees_a_duplicate_then_raises():
    engine = _engine()
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    doubled = groups + [dict(groups[0], role="dup")]
    with pytest.raises(AssertionError, match="two optimizer groups"):
        verify_partition(engine, doubled)


def test_when_codebook_group_then_it_has_no_weight_decay_and_a_positive_lr():
    engine = _engine()
    groups = build_qat_param_groups(
        engine, matrix_lr=0.02, weight_decay=0.1, codebook_lr=1e-3
    )
    codebook = [g for g in groups if g["role"] == "codebook"]
    assert len(codebook) == 1
    assert codebook[0]["weight_decay"] == 0.0
    assert codebook[0]["lr"] == pytest.approx(1e-3)


def test_when_engine_then_groups_follow_the_declared_role_order():
    engine = _engine()
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    assert [g["role"] for g in groups] == [
        r for r in ROLE_ORDER if any(g["role"] == r for g in groups)
    ]


def test_when_no_sparseprop_then_partition_still_complete():
    engine = _engine(sparsity=0.0)
    groups = build_qat_param_groups(engine, matrix_lr=0.02, weight_decay=0.1)
    verify_partition(engine, groups)
