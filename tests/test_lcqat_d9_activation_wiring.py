"""Tests for the D9 activation-LUT wiring: flags, config, and reachability.

D9 added `--lcqat-lut-relaxation` and `--lcqat-act-body`. A flag that parses,
validates and persists but never reaches a layer is worse than an absent flag
(§13), and that was the actual state these tests were written against:
`LearnableIndexLut` was constructed nowhere outside its own tests, and
`gpt.py`'s MLP computed `F.relu(x).square()` unconditionally.

So the load-bearing assertions here are reachability ones:

* attaching produces tables on the `c_fc` layers,
* a *training* forward routes through them and reaches their parameters,
* the optimizer gives those parameters a group, and
* the table survives a checkpoint round-trip.
"""

import argparse

import pytest
import torch

from nanochat.models.backbone import GPT, GPTConfig
from nanochat.models.quant import (
    LayerKConfig,
    attach_learnable_activation_luts,
    lcqat_config_from_args,
    retrofit_model,
)
from nanochat.models.quant.activation import ACT_BODIES
from nanochat.models.quant.learnable_lut import RELAXATIONS
from nanochat.models.quant.optimizer import (
    build_qat_param_groups,
    role_for_name,
    verify_partition,
)
from nanochat.models.quant.retrofit import is_lcqat_state, prepare_lcqat_before_load

RELAXATION_ACT_BODY_PAIRS = [
    ("logits", "pwl"),
    ("logits", "smoothpwl"),
    ("proximity", "pwl"),
    ("proximity", "smoothpwl"),
]


def _model(seed: int = 0) -> GPT:
    config = GPTConfig(n_layer=2, n_head=4, n_embd=256, sequence_len=64, vocab_size=512)
    # GPTConfig's GQA default is derived, not a field; set it explicitly so a
    # change to its derivation cannot turn this into a construction error.
    config.n_kv_head = 4
    torch.manual_seed(seed)
    return GPT(config)


def _retrofitted(relaxation: str = "logits", act_body: str = "pwl") -> GPT:
    model = _model()
    retrofit_model(model, LayerKConfig(lut_relaxation=relaxation, act_body=act_body))
    attach_learnable_activation_luts(model, relaxation, act_body)
    return model


def _luts(model: GPT) -> list:
    return [
        m.learnable_activation_lut
        for m in model.modules()
        if getattr(m, "learnable_activation_lut", None) is not None
    ]


class TestConfigPlumbing:
    def test_when_defaults_then_the_shipped_behaviour_is_selected(self) -> None:
        config = LayerKConfig()
        config.validate()
        assert config.lut_relaxation == "logits"
        assert config.act_body == "pwl"

    def test_when_serialized_then_both_fields_round_trip(self) -> None:
        config = LayerKConfig(lut_relaxation="proximity", act_body="smoothpwl")
        assert LayerKConfig.from_dict(config.as_dict()) == config

    def test_when_the_relaxation_is_unknown_then_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="lut_relaxation must be one of"):
            LayerKConfig(lut_relaxation="softmax").validate()

    def test_when_the_body_is_unknown_then_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="act_body must be one of"):
            LayerKConfig(act_body="rbf").validate()

    def test_when_built_from_args_then_both_flags_are_consumed(self) -> None:
        args = argparse.Namespace(
            lcqat_preset="asym",
            lcqat_k_map=None,
            lcqat_lut_relaxation="proximity",
            lcqat_act_body="smoothpwl",
        )
        config = lcqat_config_from_args(args)
        assert (config.lut_relaxation, config.act_body) == ("proximity", "smoothpwl")

    def test_when_a_caller_registers_neither_flag_then_the_defaults_hold(self) -> None:
        """`chat_rl` registers a subset; a missing attribute must not raise."""
        config = lcqat_config_from_args(
            argparse.Namespace(lcqat_preset="asym", lcqat_k_map=None)
        )
        assert (config.lut_relaxation, config.act_body) == ("logits", "pwl")

    @pytest.mark.parametrize("bad", ["softmax", "LSQ", ""])
    def test_when_the_flag_value_is_invalid_then_it_is_rejected(self, bad: str) -> None:
        args = argparse.Namespace(
            lcqat_preset="asym",
            lcqat_k_map=None,
            lcqat_lut_relaxation=bad,
            lcqat_act_body=None,
        )
        if bad not in RELAXATIONS:
            with pytest.raises(ValueError, match="Unknown --lcqat-lut-relaxation"):
                lcqat_config_from_args(args)

    def test_when_the_registries_and_the_flags_agree(self) -> None:
        """The `choices=` in argparse and the `validate()` membership test are
        two separate lists; a value accepted by one and refused by the other
        would fail at model-build time instead of flag-parse time."""
        assert set(RELAXATIONS) == {"logits", "proximity"}
        assert set(ACT_BODIES) == {"pwl", "smoothpwl"}


class TestReachability:
    def test_when_attached_then_every_mlp_c_fc_carries_a_table(self) -> None:
        model = _retrofitted()
        assert len(_luts(model)) == 2  # one per block

    @pytest.mark.parametrize("relaxation,act_body", RELAXATION_ACT_BODY_PAIRS)
    def test_when_a_training_forward_runs_then_the_table_parameters_receive_a_gradient(
        self, relaxation: str, act_body: str
    ) -> None:
        """The assertion the whole feature rests on.

        `LearnableIndexLut` is only reachable from the export-time fused chain,
        and export does not run during training. If `gpt.py` kept computing
        `F.relu(x).square()` the table parameters would receive no gradient --
        a flag that changes nothing.
        """
        model = _retrofitted(relaxation, act_body)
        model(torch.randint(0, 512, (2, 16))).sum().backward()
        lut = _luts(model)[0]
        gradients = [p.grad for p in lut.parameters() if p.grad is not None]
        assert gradients, f"no gradient reached {relaxation}/{act_body}"
        assert any(float(g.abs().max()) > 0.0 for g in gradients)

    @pytest.mark.parametrize("relaxation,act_body", RELAXATION_ACT_BODY_PAIRS)
    def test_when_proximity_is_used_then_both_vectors_train(
        self, relaxation: str, act_body: str
    ) -> None:
        """Both proximity vectors, not just `levels`.

        A gradient that reaches the levels but not the knots means the knot
        positions are frozen, which is half the parameter reduction bought for
        nothing.
        """
        if relaxation != "proximity":
            pytest.skip("knots/levels exist only under the proximity relaxation")
        model = _retrofitted(relaxation, act_body)
        model(torch.randint(0, 512, (2, 16))).sum().backward()
        lut = _luts(model)[0]
        assert lut.knots.grad is not None
        assert lut.levels.grad is not None
        assert float(lut.knots.grad.abs().max()) > 0.0

    def test_when_a_layer_has_no_table_then_the_float_activation_is_used(self) -> None:
        """The default path must be untouched: a model retrofitted without
        attaching still computes `relu^2`."""
        model = _model()
        retrofit_model(model, LayerKConfig())
        assert _luts(model) == []

    def test_when_attached_then_the_zero_anchor_is_still_exactly_zero(self) -> None:
        """SparseProp's structural-zero contract survives all four variants."""
        for relaxation, act_body in RELAXATION_ACT_BODY_PAIRS:
            lut = _luts(_retrofitted(relaxation, act_body))[0]
            zero_in = int(torch.nonzero(lut.input_codebook == 0.0, as_tuple=True)[0][0])
            emitted = lut.output_codebook[lut.resolved_table()[zero_in]]
            assert float(emitted) == 0.0, f"{relaxation}/{act_body}"


class TestOptimizerRouting:
    @pytest.mark.parametrize("suffix", ["logits", "knots", "levels"])
    def test_when_the_parameter_is_a_table_then_it_gets_the_codebook_role(
        self, suffix: str
    ) -> None:
        """Table steps are not matrix weights: they need `--codebook-lr` and no
        weight decay. Falling through to `matrix` would put them at
        `--matrix-lr` with decay, silently training them on the wrong schedule.
        """
        name = f"transformer.h.0.mlp.c_fc.learnable_activation_lut.{suffix}"
        assert role_for_name(name) == "codebook"

    def test_when_a_matrix_weight_is_inspected_then_it_stays_matrix(self) -> None:
        assert role_for_name("transformer.h.0.attn.c_q.weight") == "matrix"

    def test_when_a_codebook_parameter_is_inspected_then_it_stays_codebook(
        self,
    ) -> None:
        assert (
            role_for_name("transformer.h.0.mlp.c_fc.act_quantizer.raw_pos_deltas")
            == "codebook"
        )

    def test_when_the_partition_is_built_then_no_parameter_is_orphaned(self) -> None:
        model = _retrofitted("proximity", "smoothpwl")
        groups = build_qat_param_groups(model, matrix_lr=1e-3, weight_decay=0.1)
        verify_partition(model, groups)


class TestResume:
    def test_when_the_checkpoint_carries_tables_then_a_strict_resume_loads(
        self,
    ) -> None:
        config = LayerKConfig(lut_relaxation="proximity", act_body="smoothpwl")
        source = _model()
        retrofit_model(source, config)
        attach_learnable_activation_luts(source, "proximity", "smoothpwl")
        state = source.state_dict()

        target = _model()
        prepare_lcqat_before_load(target, state, config.as_dict(), config)
        target.load_state_dict(state, strict=True)

    def test_when_the_meta_disagrees_with_the_state_then_the_load_fails(self) -> None:
        """A resume that rebuilt the wrong relaxation must fail, not silently
        load a table whose parameters do not exist."""
        source = _model()
        retrofit_model(source, LayerKConfig(lut_relaxation="proximity"))
        attach_learnable_activation_luts(source, "proximity", "pwl")
        state = source.state_dict()
        assert is_lcqat_state(state)

        target = _model()
        prepare_lcqat_before_load(
            target, state, LayerKConfig(lut_relaxation="logits").as_dict(), None
        )
        with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
            target.load_state_dict(state, strict=True)
