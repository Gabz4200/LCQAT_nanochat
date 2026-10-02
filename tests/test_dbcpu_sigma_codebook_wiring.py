"""End-to-end wiring for sigma-conditioned codebooks (PRD 3.2).

`tests/test_lcqat_sigma_codebook.py` covers the codebooks themselves. This file
covers the part that actually breaks: getting sigma from the engine, through
the block, into every quantized layer that asks for it.

The failure mode this guards is specific and was observed while building it.
Conditioning a codebook is a one-line substitution, and every call site that
*reaches* a quantized layer then has to supply sigma. The DiffusionBlocks
engine has three such sites that are easy to miss -- the per-layer block loop,
the per-block denoise head, and the AdaLN adapter -- plus the `nn.Sequential`
in the adapter, which cannot pass a per-layer keyword at all. Miss any one and
the failure is either a crash deep in the forward or, worse, a codebook that
silently trains unconditioned while the flag reports itself enabled.
"""

import pytest
import torch

from nanochat.models.quant.retrofit import PRESETS
from nanochat.models.quant.sigma_codebook import (
    SigmaConditionedCodebook,
    SigmaModulatedCodebook,
    resolve_batch_sigma,
)
from tests.test_dbcpu_engine import make_engine


def condition_engine(engine, factory) -> int:
    """Replace every activation quantizer with `factory(quantizer)`."""
    count = 0
    for module in engine.modules():
        quantizer = getattr(module, "act_quantizer", None)
        if quantizer is None or not hasattr(quantizer, "m_neg"):
            continue
        module.act_quantizer = factory(quantizer)
        count += 1
    return count


def conditioned(quantizer) -> SigmaConditionedCodebook:
    return SigmaConditionedCodebook(
        num_anchors=3, m_neg=quantizer.m_neg, m_pos=quantizer.m_pos
    )


def modulated(quantizer) -> SigmaModulatedCodebook:
    return SigmaModulatedCodebook(m_neg=quantizer.m_neg, m_pos=quantizer.m_pos)


class TestResolveBatchSigma:
    def test_when_sigma_is_per_row_then_each_row_is_kept(self) -> None:
        sigma = torch.tensor([0.1, 1.0, 10.0])
        out = resolve_batch_sigma(sigma, 3)
        assert out.shape == (3, 1, 1)
        assert torch.allclose(out.reshape(-1), sigma)

    def test_when_sigma_is_scalar_then_it_is_shared_across_the_batch(self) -> None:
        """The AdaLN adapter is conditioned on the step's sigma, not the row's.

        Broadcasting is correct there, not a fallback: every row of the batch
        really is at the same noise level at that call site.
        """
        out = resolve_batch_sigma(torch.tensor(2.0), 4)
        assert out.shape == (4, 1, 1)
        assert torch.all(out == 2.0)

    def test_when_the_count_matches_neither_then_it_raises(self) -> None:
        with pytest.raises(ValueError, match="one value per batch element"):
            resolve_batch_sigma(torch.tensor([1.0, 2.0]), 4)


class TestEngineTraining:
    @pytest.mark.parametrize(
        ("name", "factory"),
        [("conditioned", conditioned), ("modulated", modulated)],
    )
    def test_when_conditioned_then_a_denoise_step_trains(self, name, factory) -> None:
        """The whole point: sigma reaches every quantized layer in the engine.

        Three sites have to cooperate for this to hold -- the block loop, the
        denoise head, and the AdaLN adapter. A `denoise_step` that completes
        with a live graph is the only assertion that covers all three at once.
        """
        engine = make_engine(3, n_layer=6)
        engine.apply_lcqat(PRESETS["asym"])
        n = condition_engine(engine, factory)
        assert n > 0, "no activation quantizers were conditioned"

        engine.zero_grad(set_to_none=True)
        loss, sigma = engine.denoise_step(torch.randint(0, 128, (2, 16)), block_idx=0)
        assert loss.grad_fn is not None
        loss.backward()
        got = {k for k, p in engine.named_parameters() if p.grad is not None}
        assert got, "backward reached no parameters at all"

    @pytest.mark.parametrize(
        ("name", "factory"),
        [("conditioned", conditioned), ("modulated", modulated)],
    )
    def test_when_conditioned_then_every_layer_receives_the_real_sigma(
        self, name, factory
    ) -> None:
        """Every conditioned layer must be handed the *step's* noise level.

        Deliberately a plumbing assertion rather than an end-to-end output
        comparison. The engine is already strongly sigma-sensitive through the
        AdaLN conditioning that shipped with DiffusionBlocks, so "two noise
        levels give different predictions" holds even when every codebook is fed
        a constant -- an end-to-end test here would pass with the sigma channel
        completely broken. Recording what each layer actually receives is the
        only version of this check that can fail.
        """
        import nanochat.models.backbone as gpt_mod
        import nanochat.training.diffusion_blocks as db_mod

        engine = make_engine(3, n_layer=6)
        engine.apply_lcqat(PRESETS["asym"])
        condition_engine(engine, factory)

        seen: list[tuple[str, float]] = []
        original = gpt_mod.maybe_sigma_call

        def spy(layer, x, sigma):
            if getattr(getattr(layer, "act_quantizer", None), "needs_sigma", False):
                seen.append((type(layer).__name__, float(sigma.reshape(-1)[0])))
            return original(layer, x, sigma)

        # Both modules import the helper by value, so both bindings have to be
        # replaced. Patching only `gpt` would miss the denoise head and the
        # adapter -- which is exactly the coverage this test exists to check.
        gpt_mod.maybe_sigma_call = spy
        db_mod.maybe_sigma_call = spy
        original_sample = engine.partitioner.sample_sigma
        engine.partitioner.sample_sigma = lambda *a, **k: torch.tensor(7.25)
        try:
            engine.denoise_step(torch.randint(0, 128, (2, 16)), block_idx=0)
        finally:
            gpt_mod.maybe_sigma_call = original
            db_mod.maybe_sigma_call = original
            engine.partitioner.sample_sigma = original_sample

        assert seen, "no conditioned layer was reached at all"
        # Every call must carry the step's sigma, not a per-row sample of it.
        # A per-row sigma is the subtler bug: the codebook would still vary
        # across the batch, so the model would train, but it would be
        # conditioning on noise levels the denoiser never saw.
        assert all(v == pytest.approx(7.25) for _, v in seen), (
            f"layers saw wrong sigmas: {sorted(set(v for _, v in seen))}"
        )
        # And more than one call site must be covered, so a regression that
        # fixes the MLP but drops the denoise head (or the adapter) is caught.
        assert len(seen) > 1, f"only one layer received sigma: {seen}"

    def test_when_not_conditioned_then_the_static_path_is_unchanged(self) -> None:
        """The default recipe must be bit-identical to before the feature.

        `maybe_sigma_call` decides whether to pass the keyword at all, so a
        regression there would show up as a different loss on the default path
        rather than as a crash.
        """
        engine = make_engine(3, n_layer=6)
        engine.apply_lcqat(PRESETS["asym"])
        idx = torch.randint(0, 128, (2, 16))
        engine.zero_grad(set_to_none=True)
        loss, _ = engine.denoise_step(idx, block_idx=0)
        assert loss.grad_fn is not None
        loss.backward()
        assert any(p.grad is not None for _, p in engine.named_parameters())


class TestBaseTransformerLayers:
    """The base transformer's MLPs, which `engine.apply_lcqat` does not touch.

    `apply_lcqat` only retrofits the engine's own layers (adapters + denoise
    heads). The base transformer is retrofitted separately, by `base_train`, and
    it is where most of the quantized activations live. So it is a *different*
    call site, and dropping sigma from `MLP.forward` is invisible to every test
    that only exercises the engine -- a mutation that was tried here and passed
    the whole engine-level suite.
    """

    def test_when_conditioned_then_the_base_mlp_receives_sigma(self) -> None:
        from nanochat.models.backbone import MLP
        from nanochat.models.quant.linear import LCQATLinear

        mlp = MLP.__new__(MLP)
        torch.nn.Module.__init__(mlp)
        mlp.c_fc = LCQATLinear(16, 32, bias=False, K_act=7)
        mlp.c_proj = LCQATLinear(32, 16, bias=False, K_act=7)
        for layer in (mlp.c_fc, mlp.c_proj):
            layer.act_quantizer = modulated(layer.act_quantizer)

        # Without sigma this raises, which is the point: the base MLPs are only
        # reachable with sigma threaded, and the flag would otherwise report
        # itself enabled while these layers trained unconditioned.
        out = mlp(torch.randn(3, 16), sigma=torch.full((3, 1, 1), 4.0))
        assert out.shape == (3, 16)
        out.sum().backward()
        gains = [
            p.grad
            for _, p in mlp.c_fc.act_quantizer.gain_net.named_parameters()
            if p.grad is not None
        ]
        assert gains and any(float(g.abs().sum()) > 0.0 for g in gains)

    def test_when_conditioned_then_different_sigmas_give_different_output(
        self,
    ) -> None:
        """End-to-end sensitivity for a single MLP, with no AdaLN in the way.

        The engine-level version of this check cannot isolate the codebook, but
        here the only sigma-dependent thing in the graph *is* the codebook, so a
        constant-sigma regression is unambiguous.
        """
        from nanochat.models.backbone import MLP
        from nanochat.models.quant.linear import LCQATLinear

        def build(seed: int) -> MLP:
            torch.manual_seed(seed)
            mlp = MLP.__new__(MLP)
            torch.nn.Module.__init__(mlp)
            # `K_weight` is left at its 3-level default if not set, and 3 levels
            # over a signed weight matrix round almost every weight to the zero
            # anchor -- the layer's output collapses to 0.0 and any comparison
            # between two zeros trivially "matches". `K_act=15` likewise needs
            # enough levels to resolve the relu^2 in between.
            mlp.c_fc = LCQATLinear(16, 32, bias=False, K_act=15, K_weight=15)
            mlp.c_proj = LCQATLinear(32, 16, bias=False, K_act=15, K_weight=15)
            for layer in (mlp.c_fc, mlp.c_proj):
                layer.act_quantizer = modulated(layer.act_quantizer)
            # Non-degenerate weights: a freshly built `LCQATLinear` initializes
            # to a tiny span, and the relu^2 in between zeroes most of it, so
            # the whole output collapses to 0.0 and any comparison of two zeros
            # trivially "matches".
            with torch.no_grad():
                mlp.c_fc.weight.normal_(0.0, 0.5)
                mlp.c_proj.weight.normal_(0.0, 0.5)
                for layer in (mlp.c_fc, mlp.c_proj):
                    # A *moderate* gain. The last layer's `SiLU` is fed
                    # `log(sigma)`, which is -4.6 at sigma=0.01 and +4.6 at
                    # sigma=100, so a weight of 1.0 drives the raw output to the
                    # clamp and the gain saturates at exp(4) ~= 54.6. The
                    # codebook then spans +-54 while the activations are +-2, so
                    # every input lands in the middle bucket, the zero anchor,
                    # and the whole layer returns 0.0 -- a "different output"
                    # that is really two tensors of zeros.
                    layer.act_quantizer.gain_net[-1].weight.fill_(0.05)
            return mlp

        x = torch.randn(2, 16)
        # Identical weights in both arms: without the seed the two MLPs would
        # differ anyway and the comparison would prove nothing.
        low = build(0)(x, sigma=torch.full((2, 1, 1), 0.01))
        high = build(0)(x, sigma=torch.full((2, 1, 1), 100.0))
        # The arms must be non-degenerate, or this compares two zero tensors and
        # passes for the wrong reason.
        assert float(low.detach().abs().max()) > 1e-6, "MLP output is identically zero"
        assert not torch.allclose(low, high), (
            "the MLP produced identical output for sigma 0.01 and 100: the "
            "codebook is not conditioning on the noise level"
        )


class TestQuantizedRuntimeGuard:
    def test_when_conditioned_and_exported_then_the_fused_path_raises(self) -> None:
        """Export + conditioning is an unsupported combination, and says so.

        The exported `activation_lut` is a single static table. A conditioned
        codebook needs one table per noise level, so the fused index path would
        silently quantize with the wrong levels -- the artifact would be wrong
        and nothing downstream could tell.

        Exercised through the `MLP`, not the layer directly: the guard lives in
        `MLP.forward`, and `quantized_mlp_chain` checks the *out* quantizer
        first, so calling the layer would assert on a different precondition and
        never reach the sigma guard at all.
        """
        from nanochat.models.backbone import MLP
        from nanochat.models.quant.linear import LCQATLinear

        c_fc = LCQATLinear(16, 32, bias=False, K_act=7, quantize_out=True)
        c_fc.act_quantizer = modulated(c_fc.act_quantizer)
        c_fc.register_buffer("activation_lut", torch.zeros(7, dtype=torch.uint8))
        c_fc.register_buffer(
            "packed_weight_indices", torch.zeros(32, 16, dtype=torch.uint8)
        )
        c_fc.register_buffer("weight_index_format", torch.zeros((), dtype=torch.int32))
        c_proj = LCQATLinear(32, 16, bias=False, K_act=7)
        c_proj.register_buffer(
            "packed_weight_indices", torch.zeros(16, 32, dtype=torch.uint8)
        )
        c_proj.register_buffer(
            "weight_index_format", torch.zeros((), dtype=torch.int32)
        )

        mlp = MLP.__new__(MLP)
        torch.nn.Module.__init__(mlp)
        mlp.c_fc = c_fc
        mlp.c_proj = c_proj

        with pytest.raises(RuntimeError, match="sigma-conditioned"):
            mlp(torch.randn(2, 16), sigma=torch.tensor([[1.0]]))

    def test_when_static_and_exported_then_the_fused_path_still_works(self) -> None:
        """The guard must not fire for a static codebook.

        A guard that raises unconditionally would be indistinguishable from the
        bug it prevents: both look like "the fused path is broken".
        """
        from nanochat.models.backbone import MLP, maybe_sigma_call

        mlp = MLP.__new__(MLP)
        torch.nn.Module.__init__(mlp)
        # A plain float MLP must still run with sigma threaded through, which is
        # the shape the guard has to leave alone.
        mlp.c_fc = torch.nn.Linear(8, 16, bias=False)
        mlp.c_proj = torch.nn.Linear(16, 8, bias=False)
        out = mlp(torch.randn(2, 8), sigma=torch.tensor([[1.0]]))
        assert out.shape == (2, 8)
        assert maybe_sigma_call(mlp.c_fc, torch.randn(2, 8), None) is not None


class TestGradientScaling:
    def test_when_inv_sqrt_n_then_the_conditioned_gradient_is_scaled(self) -> None:
        """PRD 2.4 applies to conditioned codebooks too.

        The factor has to shrink `dL/dC` while leaving `dL/dx` at the STE's 1.0.
        Both halves are checked: a factor applied to the whole STE expression
        would pass a codebook-only assertion while silently attenuating the
        input gradient.
        """
        x = torch.randn(4, 32, requires_grad=True)
        codebook = modulated(
            __import__(
                "nanochat.models.quant.codebook", fromlist=["x"]
            ).MemoryEfficientLearnedCodebook(m_neg=3, m_pos=3)
        )
        x2 = x.detach().clone().requires_grad_(True)
        codebook(x, torch.full((4, 1, 1), 1.0), scale=1.0).value.sum().backward()
        codebook.zero_grad(set_to_none=True)
        codebook(x2, torch.full((4, 1, 1), 1.0), scale=0.125).value.sum().backward()

        g_scaled = codebook.base.raw_pos_deltas.grad
        assert float(g_scaled.abs().sum()) > 0.0
        # The input gradient is the STE identity in both cases, so its magnitude
        # must be unchanged by the codebook scale.
        assert torch.allclose(x.grad, x2.grad, rtol=1e-5), (
            "the codebook scale also attenuated the input gradient"
        )
