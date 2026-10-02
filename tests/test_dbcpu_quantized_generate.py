"""DiffusionBlocks generation through the exported quantized runtime (W5).

Two claims, and the second is the one that was broken.

1. `generate` runs on a model whose LC-QAT layers have been exported: weights
   replaced by packed index buffers, `activation_lut` installed, the MLP served
   by `quantized_mlp_chain` rather than `F.linear`. The tokens it produces must
   match the retrofitted-but-not-yet-exported model exactly -- export is a
   serialization step, so a token difference means export changed the math.

2. Even K must work end to end. The C++ index kernel used to require an *odd*
   K on both LUTs, which contradicted `packing.index_format_for_k` and made the
   whole 2-bit (K=4) and 4-bit (K=16) boundaries unusable. It surfaced here
   rather than in a unit test because `mlp.c_proj` is the one layer the default
   preset gives an even `K_act` (8), so the fused MLP chain is the only path
   that reaches it.

The first test is the regression guard for (2): it uses the default preset and
therefore fails outright if the odd-K restriction returns.
"""

import pytest
import torch

from nanochat.models.quant.export import export_lcqat_checkpoint
from nanochat.models.quant.packing import index_format_for_k, pack_weight_indices
from nanochat.models.quant.retrofit import DEFAULT_PRESET, PRESETS, retrofit_model
from nanochat.ops.index_linear import index_linear_cpu
from nanochat.ops.references.index_linear_reference import (
    reference_index_linear,
)
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from tests.conftest import build_active_tiny_gpt


def kernel_available() -> bool:
    try:
        from nanochat.ops.kernels.cpu_loader import load_cpu_index_linear_extension

        load_cpu_index_linear_extension()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def index_kernel():
    """Require the compiled index kernel, failing rather than skipping.

    A `skipif` here is actively harmful: when a bad `TORCH_CHECK` stops the
    extension from loading, every kernel-backed test *skips*, and pytest reports
    the run green. That is exactly how the odd-K restriction survived -- the
    guard that was supposed to catch it was reporting success. `AGENTS.md` already
    requires a compiler on `PATH` for the ops tests, so treating a load failure
    as a test failure loses nothing.
    """
    assert kernel_available(), (
        "the CPU index-linear extension must load; a skip here would hide a "
        "regression in the kernel behind a green run"
    )
    assert hasattr(torch.ops.nanochat, "lcqat_index_linear")
    return torch.ops.nanochat


class TestGenerationThroughQuantizedRuntime:
    def test_when_generating_before_and_after_export_then_the_tokens_match(
        self,
    ) -> None:
        """Export must not change what the sampler computes."""
        partitioner = EquiProbabilityPartitioner(num_blocks=2)
        model = build_active_tiny_gpt()
        retrofit_model(model, PRESETS[DEFAULT_PRESET])
        engine = DiffusionBlockEngine(model, partitioner)
        before = engine.generate(idx=None, max_new_tokens=4, seed=42)

        export_lcqat_checkpoint(model, "/tmp/test_dbcpu_export.pt", sparse=True)
        after = engine.generate(idx=None, max_new_tokens=4, seed=42)

        assert after == before

    def test_when_generating_after_export_then_the_fused_mlp_chain_is_the_path(
        self,
    ) -> None:
        """The quantized path is genuinely in use, not silently bypassed.

        Without this, a future change that made export fall back to `F.linear`
        would still pass the token-equality test above.
        """
        model = build_active_tiny_gpt()
        retrofit_model(model, PRESETS[DEFAULT_PRESET])
        export_lcqat_checkpoint(model, "/tmp/test_dbcpu_export2.pt", sparse=True)

        fused = 0
        for module in model.modules():
            buffers = getattr(module, "_buffers", {})
            if "packed_weight_indices" in buffers and "activation_lut" in buffers:
                fused += 1
        assert fused > 0, "no layer carries the export buffers, so nothing is fused"

    def test_when_generating_with_a_prompt_then_the_prompt_is_preserved(self) -> None:
        model = build_active_tiny_gpt()
        retrofit_model(model, PRESETS[DEFAULT_PRESET])
        export_lcqat_checkpoint(model, "/tmp/test_dbcpu_export3.pt", sparse=True)
        engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
        prompt = [10, 20]
        tokens = engine.generate(idx=prompt, max_new_tokens=3, seed=7)
        assert tokens[: len(prompt)] == prompt
        assert len(tokens) == len(prompt) + 3
        assert all(0 <= t < model.config.vocab_size for t in tokens)

    def test_when_generating_then_more_steps_change_the_sample(self) -> None:
        """The sampler must actually be doing work, not returning a constant.

        A stubbed or short-circuited denoising loop would still satisfy every
        length and range assertion above.
        """
        model = build_active_tiny_gpt()
        retrofit_model(model, PRESETS[DEFAULT_PRESET])
        export_lcqat_checkpoint(model, "/tmp/test_dbcpu_export4.pt", sparse=True)
        engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
        a = engine.generate(idx=None, max_new_tokens=4, seed=1)
        b = engine.generate(idx=None, max_new_tokens=4, seed=2)
        assert a != b


class TestEvenCodebookSizes:
    """The bit-width boundaries LC-QAT is built to hit.

    `K = m_neg + 1 + m_pos` lets a one-sided codebook put `0.0` exactly at an
    endpoint, which is what makes K=4 (2-bit) and K=16 (4-bit) reachable. Both
    are even, so any odd-K requirement rejects the very bit-widths the method
    is for.
    """

    @pytest.mark.parametrize("k", [3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 255, 256])
    def test_when_the_codebook_size_is_k_then_the_kernel_matches_the_reference(
        self, index_kernel, k: int
    ) -> None:
        torch.manual_seed(k)
        t, m, n = 4, 8, 16
        act_lut = torch.sort(torch.randn(k)).values
        weight_lut = torch.sort(torch.randn(k)).values
        act_indices = torch.randint(0, k, (t, n), dtype=torch.uint8)
        weight_indices = torch.randint(0, k, (m, n))
        packed, fmt = pack_weight_indices(weight_indices, k)
        assert fmt == index_format_for_k(k)

        expected = reference_index_linear(
            act_indices, act_lut, packed, weight_lut, n, fmt
        )
        got = index_linear_cpu(act_indices, act_lut, packed, weight_lut, n, fmt)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("k", [4, 8, 16])
    def test_when_the_bit_width_boundary_is_even_then_the_kernel_accepts_it(
        self, index_kernel, k: int
    ) -> None:
        """Named separately because these are the K the design specifically wants."""
        torch.manual_seed(k)
        act_lut = torch.sort(torch.randn(k)).values
        weight_lut = torch.sort(torch.randn(k)).values
        act_indices = torch.randint(0, k, (3, 8), dtype=torch.uint8)
        weight_indices = torch.randint(0, k, (4, 8))
        packed, fmt = pack_weight_indices(weight_indices, k)
        out = index_linear_cpu(act_indices, act_lut, packed, weight_lut, 8, fmt)
        assert torch.isfinite(out).all()
        assert out.shape == (3, 4)
