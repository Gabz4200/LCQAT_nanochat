"""`GPT.forward`'s value-embedding lookup must survive a depth change.

`value_embeds` is keyed by layer index as a string. The obvious optimisation is
to precompute the lookup, but the keys are not static: `has_ve(layer_idx,
n_layer)` depends on the total depth, so `resize_to` replaces the whole
`ModuleDict` when the depth changes and rebuilds the per-layer vectors.

A cache built in `__init__` therefore goes stale, and a list indexed by
`range(config.n_layer)` raises `IndexError` when `transformer.h` ends up longer
than the config's `n_layer` -- which is exactly what the experiment harness
produces when it rebuilds a model at a different depth. The membership form
degrades to `ve=None`, matching how every other mismatch in that loop behaves.

So the lookup is derived from the live `ModuleDict` on each forward.
"""

import pytest
import torch

from nanochat.modules.experiments.tiny_models import (
    build_active_tiny_gpt,
    resize_to,
)


def _logits(model, idx):
    model.eval()
    with torch.no_grad():
        return model(idx)


class TestValueEmbedLookupSurvivesResize:
    def test_the_lookup_matches_a_direct_indexed_read(self):
        """The optimisation must be a pure lookup, not a behaviour change."""
        model = build_active_tiny_gpt()
        idx = torch.randint(0, 128, (1, 8))

        # Reference: what the old inline form computes, layer by layer.
        reference = []
        for i in range(len(model.transformer.h)):
            reference.append(
                model.value_embeds[str(i)](idx).to(torch.float32)
                if str(i) in model.value_embeds
                else None
            )

        ve_by_layer = {int(k): v for k, v in model.value_embeds.items() if k.isdigit()}
        for i in range(len(model.transformer.h)):
            got = ve_by_layer[i](idx).to(torch.float32) if i in ve_by_layer else None
            want = reference[i]
            if want is None:
                assert got is None
            else:
                torch.testing.assert_close(got, want)

    def test_the_model_still_runs_after_a_depth_change(self):
        """The regression: an IndexError here broke every post-resize forward."""
        model = build_active_tiny_gpt()
        idx = torch.randint(0, 128, (1, 8))
        before = _logits(model, idx)
        assert torch.isfinite(before).all()

        resize_to(model, 4)
        after = _logits(model, idx)
        assert torch.isfinite(after).all()

    def test_a_doubled_depth_runs(self):
        """`transformer.h` longer than the config's n_layer must not raise."""
        model = build_active_tiny_gpt()
        resize_to(model, 8)
        assert len(model.transformer.h) == 8
        out = _logits(model, torch.randint(0, 128, (1, 8)))
        assert torch.isfinite(out).all()

    def test_resizing_moves_the_value_embeddings_rather_than_freezing_them(self):
        """`has_ve` depends on depth, so which layers carry one must change."""
        model = build_active_tiny_gpt()
        before = sorted(int(k) for k in model.value_embeds.keys() if k.isdigit())

        resize_to(model, 4)
        after = sorted(int(k) for k in model.value_embeds.keys() if k.isdigit())

        assert before != after, "the resize did not move any value embedding"
        assert _logits(model, torch.randint(0, 128, (1, 8))) is not None

    def test_the_gate_is_symmetric_with_the_ve_gate(self):
        """A value embedding without its gate crashes in `block`, so they must agree."""
        model = build_active_tiny_gpt()
        for n_layer in (2, 4, 6):
            resize_to(model, n_layer)
            for i, block in enumerate(model.transformer.h):
                has_ve = str(i) in model.value_embeds
                assert has_ve == (block.attn.ve_gate is not None), (
                    f"depth {n_layer}, layer {i}: value embedding and gate disagree"
                )
            assert _logits(model, torch.randint(0, 128, (1, 8))) is not None


# Depth 1 is excluded: `resize_to` shrinks by truncating, and at depth 1 the one
# surviving block came from a depth-2 model where layer 0 carries no value
# embedding, so there is no `ve_gate` to copy the shape from. That is a
# limitation of the resize *helper*, not of the forward pass -- a GPT built
# natively at depth 1 is fine. Widening `resize_to` to synthesize a gate is out
# of scope here.
@pytest.mark.parametrize("n_layer", [2, 3, 4, 5, 6])
def test_every_depth_forward_runs(n_layer):
    model = build_active_tiny_gpt()
    resize_to(model, n_layer)
    out = _logits(model, torch.randint(0, 128, (1, 8)))
    assert torch.isfinite(out).all()
