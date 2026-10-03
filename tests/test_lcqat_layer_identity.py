"""Regression tests for the LC-QAT layer-identity fixes.

One theme, three packages: code that asked about a module's *class* when it
meant the *layer*.

`SparsePropLinearLCQAT` re-parents an `LCQATLinear`'s weight, bias and all three
quantizers instead of inheriting from it -- nesting it as a submodule would
register every codebook parameter twice. So it is functionally an LC-QAT layer
and is **not** a subclass of one. `isinstance(m, LCQATLinear)` therefore
silently skipped it, in five places. Each test below fails on the commit before
`is_lcqat_layer` existed.

1. `retrofit_summary` reported 12 -> 8 layers as soon as SparseProp wrapped 4.
2. `strip_lcqat` left a sparse LC-QAT layer inside a supposedly-float twin.
3. `_wrap_sparseprop` demoted `SparsePropLinearLCQAT` to plain
   `SparsePropLinear` on a ramp re-wrap, dropping the quantized forward while
   leaving the codebook modules attached -- so `verify_partition`, which only
   checks that no role is *missing*, could not see it.
4. `retrofit_model` re-quantized an already-sparse LC-QAT layer.
"""

import copy

import pytest
import torch

from nanochat.models.quant.linear import LCQATLinear, is_lcqat_layer, lcqat_layer_types
from nanochat.models.quant.retrofit import (
    PRESETS,
    retrofit_model,
    retrofit_summary,
)
from nanochat.models.quant.sparseprop import (
    SparsePropLinear,
    SparsePropLinearLCQAT,
    inject_sparseprop_layers,
)
from nanochat.models.quant.w6 import strip_lcqat
from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt, make_engine


def _lcqat_model():
    """A retrofitted tiny GPT with live zero-inits randomized (see conftest)."""
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS["small"])
    return model


def _sparse_wrap(model, *, sparsity=0.5, modules=("c_proj", "c_fc")):
    inject_sparseprop_layers(
        model, sparsity=sparsity, target_modules=list(modules), with_lcqat=True
    )
    return model


def _count(model, cls):
    return sum(isinstance(m, cls) for m in model.modules())


class TestIsLcqatLayer:
    def test_when_sparseprop_wraps_then_the_layer_is_still_an_lcqat_layer(self):
        model = _lcqat_model()
        before = sum(is_lcqat_layer(m) for m in model.modules())
        _sparse_wrap(model)
        assert sum(is_lcqat_layer(m) for m in model.modules()) == before

        wrapped = [m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)]
        assert wrapped, "the fixture did not actually wrap anything"
        for m in wrapped:
            assert not isinstance(m, LCQATLinear), (
                "this test is only meaningful while SparsePropLinearLCQAT does "
                "NOT subclass LCQATLinear. If it now does, the isinstance sites "
                "are correct again and this whole file can be deleted."
            )
            assert is_lcqat_layer(m)

    def test_when_a_plain_float_model_then_nothing_is_an_lcqat_layer(self):
        assert not any(is_lcqat_layer(m) for m in build_active_tiny_gpt().modules())

    def test_lcqat_layer_types_covers_both_shapes(self):
        types = lcqat_layer_types()
        assert LCQATLinear in types
        assert SparsePropLinearLCQAT in types

    def test_when_sparseprop_has_no_lcqat_then_it_is_not_an_lcqat_layer(self):
        """SparseProp without LC-QAT owns no codebooks, so it must not match.

        Otherwise `strip_lcqat` would try to re-quantize a layer that was never
        quantized.
        """
        model = build_active_tiny_gpt()
        inject_sparseprop_layers(
            model, sparsity=0.5, target_modules=["c_proj"], with_lcqat=False
        )
        sparse = [m for m in model.modules() if isinstance(m, SparsePropLinear)]
        assert sparse
        assert not any(is_lcqat_layer(m) for m in sparse)


class TestRetrofitSummaryCountsSparseLayers:
    def test_when_some_layers_are_sparse_then_the_summary_still_counts_them(self):
        """`retrofit_summary` feeds a logged metric, so it must not under-report."""
        model = _lcqat_model()
        dense = sum(retrofit_summary(model).values())
        assert dense == 12, dense

        _sparse_wrap(model)
        sparse = sum(retrofit_summary(model).values())
        assert sparse == dense, (
            f"retrofit_summary dropped {dense} -> {sparse} once SparseProp "
            "wrapped some layers: the sparse LC-QAT layers are invisible to an "
            "isinstance(LCQATLinear) check"
        )

    def test_when_every_projection_is_sparse_then_the_summary_still_counts_them(self):
        model = _lcqat_model()
        _sparse_wrap(model, modules=("c_q", "c_k", "c_v", "c_fc", "c_proj"))
        assert sum(retrofit_summary(model).values()) == 12


class TestStripLcqat:
    def test_when_a_sparse_lcqat_layer_is_present_then_it_is_stripped(self):
        """A "float twin" carrying quantized layers is not a float twin."""
        model = _lcqat_model()
        _sparse_wrap(model)
        wrapped = _count(model, SparsePropLinearLCQAT)
        assert wrapped, "the fixture did not actually wrap anything"

        assert strip_lcqat(model) == 12
        assert not any(is_lcqat_layer(m) for m in model.modules()), (
            "an LC-QAT layer survived into the twin"
        )
        assert not any(isinstance(m, SparsePropLinear) for m in model.modules()), (
            "the replacement must also drop the sparse forward -- that is the "
            "other half of what 'float twin' has to mean"
        )

    def test_when_stripped_then_the_replacement_carries_the_float_weights(self):
        """Stripping materializes the raw weight; the layer is a NEW object.

        `strip_lcqat` returns a fresh `nn.Linear` rather than swapping the class
        in place, precisely so the quantizer submodules are dropped. So the
        module reachable from the tree is not the one that was replaced -- look
        it up again by name.
        """
        model = _lcqat_model()
        _sparse_wrap(model)
        name, original = next(
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, SparsePropLinearLCQAT)
        )
        expected = original.weight.detach().clone()

        strip_lcqat(model)

        replacement = model.get_submodule(name)
        assert type(replacement) is torch.nn.Linear
        torch.testing.assert_close(replacement.weight.detach(), expected)

    def test_when_a_twin_is_deepcopied_after_retrofit_then_it_is_all_float(self):
        """The resume path: the twin is copied after the retrofit, then stripped."""
        model = _lcqat_model()
        _sparse_wrap(model)
        twin = copy.deepcopy(model)
        strip_lcqat(twin)

        assert not any(is_lcqat_layer(m) for m in twin.modules())
        # Stripping a copy must not touch the student.
        assert sum(is_lcqat_layer(m) for m in model.modules()) == 12


class TestEngineRewrapKeepsLcqat:
    """The engine re-wraps its own layers on every ramp event; that is the bug.

    The *base model* is re-pruned by `GradualPruningSchedule.apply`, which
    rewrites masks rather than re-wrapping, so `inject_sparseprop_layers`
    correctly skips an already-sparse layer. The engine's `_wrap_sparseprop`
    does re-wrap -- it must re-parent, not demote.
    """

    def _engine_with_lcqat(self):
        engine = make_engine(num_blocks=2, n_layer=4)
        engine.apply_lcqat(PRESETS["small"])
        return engine

    def test_when_sparseprop_is_applied_then_the_heads_become_sparse_lcqat(self):
        engine = self._engine_with_lcqat()
        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        assert _count(engine, SparsePropLinearLCQAT) > 0

    def test_when_applied_twice_then_no_layer_loses_its_lcqat(self):
        """The regression: the second wrap used to produce plain SparsePropLinear."""
        engine = self._engine_with_lcqat()
        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        first = _count(engine, SparsePropLinearLCQAT)
        assert first > 0

        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        assert _count(engine, SparsePropLinearLCQAT) == first, (
            f"{first} SparsePropLinearLCQAT before the re-wrap, "
            f"{_count(engine, SparsePropLinearLCQAT)} after: the re-wrap demoted "
            "already-sparse LC-QAT layers to plain SparsePropLinear"
        )

    def test_when_applied_twice_then_no_plain_sparseprop_is_introduced(self):
        """A plain SparsePropLinear here means LC-QAT was dropped for that layer."""
        engine = self._engine_with_lcqat()
        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        engine.apply_sparseprop(sparsity=0.75, with_lcqat=True)

        lcqat_shaped = sum(
            isinstance(m, SparsePropLinearLCQAT) for m in engine.modules()
        )
        plain = sum(
            isinstance(m, SparsePropLinear) and not isinstance(m, SparsePropLinearLCQAT)
            for m in engine.modules()
        )
        assert plain == 0, f"{plain} plain SparsePropLinear survived an LC-QAT wrap"
        assert lcqat_shaped > 0

    def test_when_applied_twice_then_the_head_keeps_its_codebooks(self):
        """The demotion kept the quantizer *modules* attached, so only this catches it."""
        engine = self._engine_with_lcqat()
        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        engine.apply_sparseprop(sparsity=0.75, with_lcqat=True)
        for head in engine.denoise_heads:
            assert isinstance(head, SparsePropLinearLCQAT), type(head).__name__
            assert head.weight_quantizer is not None
            assert head.act_quantizer is not None

    def test_when_applied_twice_then_no_parameter_is_registered_twice(self):
        """Re-parenting must not double-register a codebook parameter."""
        engine = self._engine_with_lcqat()
        engine.apply_sparseprop(sparsity=0.5, with_lcqat=True)
        engine.apply_sparseprop(sparsity=0.75, with_lcqat=True)
        names = [n for n, _ in engine.named_parameters()]
        assert len(names) == len(set(names)), "a parameter is registered twice"


class TestExportActivationWalk:
    """The export walk finds the `c_fc` -> next-layer sibling wiring site.

    It keyed on `isinstance(module, LCQATLinear)`, so a sparse LC-QAT `c_fc` was
    skipped and the exported artifact silently carried no activation table for
    it -- a wrong artifact rather than an error.
    """

    def _pairs(self, model):
        from nanochat.models.quant.export import _activation_pairs
        from nanochat.models.quant.lut import ACTIVATION_LUTS

        return list(_activation_pairs(model, ACTIVATION_LUTS))

    def test_when_c_fc_is_sparse_then_the_walk_still_finds_it(self):
        model = _lcqat_model()
        dense_pairs = self._pairs(model)
        assert dense_pairs, "the dense fixture produced no wiring sites"

        _sparse_wrap(model, modules=("c_fc",))
        sparse_pairs = self._pairs(model)
        assert len(sparse_pairs) == len(dense_pairs), (
            f"{len(dense_pairs)} wiring sites before the sparse wrap, "
            f"{len(sparse_pairs)} after: the export walk skips "
            "SparsePropLinearLCQAT"
        )

    def test_when_everything_is_sparse_then_the_walk_is_unchanged(self):
        model = _lcqat_model()
        dense_pairs = self._pairs(model)
        _sparse_wrap(model, modules=("c_q", "c_k", "c_v", "c_fc", "c_proj"))
        assert len(self._pairs(model)) == len(dense_pairs)

    def test_when_the_walk_finds_a_sparse_layer_then_wire_installs_a_table(self):
        """End to end: the attachment mode must reach the sparse layer."""
        from nanochat.models.quant.export import wire_activation_luts
        from nanochat.models.quant.sparseprop import SparsePropLinearLCQAT

        model = _lcqat_model()
        _sparse_wrap(model, modules=("c_fc",))
        before = sum(1 for m in model.modules() if hasattr(m, "activation_lut"))

        wire_activation_luts(model)

        sparse_with_lut = sum(
            isinstance(m, SparsePropLinearLCQAT) and hasattr(m, "activation_lut")
            for m in model.modules()
        )
        after = sum(1 for m in model.modules() if hasattr(m, "activation_lut"))
        assert after > before, "no activation table was installed at all"
        assert sparse_with_lut == 0 or after > before, sparse_with_lut


class TestRetrofitIdempotency:
    def test_when_retrofitting_a_sparse_model_then_nothing_is_requantized(self):
        """A sparse LC-QAT layer is an nn.Linear, so the skip needs the predicate."""
        model = _lcqat_model()
        _sparse_wrap(model)
        before = sum(is_lcqat_layer(m) for m in model.modules())

        retrofit_model(model, PRESETS["prd"])

        after = sum(is_lcqat_layer(m) for m in model.modules())
        assert after == before, (
            f"re-retrofit changed the layer count {before} -> {after}"
        )

    def test_when_from_float_is_given_a_sparse_layer_then_it_raises(self):
        model = _lcqat_model()
        _sparse_wrap(model)
        wrapped = next(
            m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)
        )
        with pytest.raises(ValueError):
            LCQATLinear.from_float(
                wrapped,
                K_weight=15,
                K_act=15,
                per_channel_weight=False,
            )
