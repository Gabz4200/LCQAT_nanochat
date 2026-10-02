"""Parity between the frozen and learnable activation LUT in the fused chain.

The whole point of `LearnableIndexLut` being initialised to the frozen bake is
that switching it on cannot change what the model computes at step 0. This file
tests that claim at the level where it is actually consumed -- the fused
`quantized_mlp_chain` gather -- rather than at the level of the table in
isolation.

If `_apply_activation_lut` ever silently fell back to the frozen buffer while a
learnable table was attached, the training path would optimise one activation
and inference would run another. That failure is invisible in loss and shows up
only as an accuracy gap, so it gets pinned here.
"""

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.learnable_lut import LearnableIndexLut, bake_learnable_table
from nanochat.models.quant.linear import LCQATLinear


def build_mlp(k_weight: int = 3, k_act: int = 15, quantize_out: bool = True):
    """A retrofitted c_fc -> c_proj pair with the export buffers installed."""
    torch.manual_seed(11)
    c_fc = LCQATLinear.from_float(
        nn.Linear(16, 16),
        K_weight=k_weight,
        K_act=k_act,
        quantize_out=quantize_out,
        act_init=(-3.0, 3.0),
    )
    c_proj = LCQATLinear.from_float(
        nn.Linear(16, 16), K_weight=k_weight, K_act=k_act, quantize_out=False
    )
    c_fc.register_buffer(
        "activation_lut",
        bake_learnable_table(
            c_fc.out_quantizer.get_codebook(), c_proj.act_quantizer.get_codebook()
        ).to(torch.uint8),
    )
    # The fused chain dispatches on packed buffers, so install a trivial packed
    # form for each side. Parity is about the activation table, not the matmul.
    # The reference index kernel requires uint8 act indices, so the buffer is
    # stored in the same uint8 form the real export produces for K <= 16.
    for layer in (c_fc, c_proj):
        indices = layer.weight_quantizer(layer.weight).indices
        layer.register_buffer("packed_weight_indices", indices.to(torch.uint8))
        layer.register_buffer("weight_index_format", torch.tensor(0))
    return c_fc, c_proj


class TestFusedChainLutParity:
    def test_when_a_learnable_table_is_attached_then_the_chain_output_is_unchanged(
        self,
    ) -> None:
        """At initialisation the fused chain must be bit-identical."""
        c_fc, c_proj = build_mlp()
        x = torch.randn(4, 16)
        with torch.no_grad():
            frozen = c_fc.quantized_mlp_chain(x, c_proj)

            learnable = LearnableIndexLut(
                c_fc.out_quantizer.get_codebook(),
                c_proj.act_quantizer.get_codebook(),
                "relu2",
            )
            c_fc.learnable_activation_lut = learnable
            trained_lut = c_fc.quantized_mlp_chain(x, c_proj)

        torch.testing.assert_close(frozen, trained_lut, rtol=0.0, atol=0.0)

    def test_when_the_table_is_trained_then_the_chain_changes(self) -> None:
        """The converse: if the chain ignored the table, training would be a no-op."""
        c_fc, c_proj = build_mlp()
        learnable = LearnableIndexLut(
            c_fc.out_quantizer.get_codebook(),
            c_proj.act_quantizer.get_codebook(),
            "relu2",
        )
        c_fc.learnable_activation_lut = learnable
        x = torch.randn(4, 16)
        with torch.no_grad():
            before = c_fc.quantized_mlp_chain(x, c_proj)
            # Repoint an index the chain actually produces. Index 0 is the
            # zero anchor: relu^2 maps every negative input there, so editing
            # that row would leave the output unchanged and the test would pass
            # for the wrong reason.
            table = learnable.resolved_table()
            reachable = table[table != 0]
            assert reachable.numel() > 0, "no non-zero output index to test with"
            victim = int(reachable[0])
            learnable.logits[victim, :] = -10.0
            learnable.logits[victim, (victim + 1) % learnable.k_out] = 10.0
            after = c_fc.quantized_mlp_chain(x, c_proj)

        assert not torch.allclose(before, after)

    def test_when_no_learnable_table_is_attached_then_the_frozen_buffer_is_used(
        self,
    ) -> None:
        c_fc, _ = build_mlp()
        ids = torch.randint(0, c_fc.activation_lut.numel(), (3, 5))
        expected = c_fc.activation_lut[ids.long()]
        assert torch.equal(c_fc._apply_activation_lut(ids), expected)

    def test_when_a_learnable_table_is_attached_then_it_wins_over_the_frozen_buffer(
        self,
    ) -> None:
        """Precedence, because silently preferring the buffer is the exact bug."""
        c_fc, _ = build_mlp()
        learnable = LearnableIndexLut(
            c_fc.out_quantizer.get_codebook(),
            c_fc.out_quantizer.get_codebook(),
            "relu2",
        )
        c_fc.learnable_activation_lut = learnable
        ids = torch.randint(0, learnable.k_in, (3, 5))
        expected = learnable.resolved_table()[ids.long()]
        assert torch.equal(c_fc._apply_activation_lut(ids), expected)

    @pytest.mark.parametrize("k_act", [3, 15, 16])
    def test_when_indices_flow_through_then_they_stay_in_range(
        self, k_act: int
    ) -> None:
        """Out-of-range indices would silently corrupt the following matmul."""
        c_fc, c_proj = build_mlp(k_act=k_act)
        learnable = LearnableIndexLut(
            c_fc.out_quantizer.get_codebook(),
            c_proj.act_quantizer.get_codebook(),
            "relu2",
        )
        c_fc.learnable_activation_lut = learnable
        ids = torch.randint(0, learnable.k_in, (6, 9))
        out = c_fc._apply_activation_lut(ids)
        assert int(out.min()) >= 0
        assert int(out.max()) < c_proj.act_quantizer.get_codebook().numel()
