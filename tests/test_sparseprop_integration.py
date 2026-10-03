"""Integration tests for SparsePropLinear module: autograd equivalence,
mask integrity, and DB-CPU block-wise integration."""

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.linear import LCQATLinear
from nanochat.models.quant.sparseprop import (
    SparsePropLinear,
    SparsePropLinearLCQAT,
    apply_static_sparsity_mask,
    inject_sparseprop_layers,
)


def build_tiny_gpt():
    """A depth-2 GPT with the shape this file's tests use.

    One definition, because this block was repeated four times in the file and
    the copies were not interchangeable: `retrofit_model` and
    `inject_sparseprop_layers` both mutate in place and re-parent modules, so
    every arm of a comparison needs its own instance. `init_weights()`
    randomizes nothing, but the callers that need live zero-inits must call
    `build_active_tiny_gpt` instead -- a zero `c_proj` transmits no gradient and
    every gradient assertion over it passes vacuously.
    """
    from nanochat.models.backbone import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model = model.to_empty(device="cpu")
    model.init_weights()
    return model


#: A 4x6 mask with no row or column fully zero, so every row keeps at least one
#: nnz (the per-row floor `magnitude_mask` enforces).
SAMPLE_MASK_ROWS = (
    (1, 0, 1, 0, 1, 0),
    (0, 1, 0, 1, 0, 1),
    (1, 1, 1, 0, 0, 1),
    (0, 0, 1, 1, 1, 0),
)


def sample_mask() -> torch.Tensor:
    """A fresh copy of `SAMPLE_MASK_ROWS` as a bool tensor."""
    return torch.tensor(SAMPLE_MASK_ROWS, dtype=torch.bool)


@pytest.fixture
def tiny_model():
    model = build_tiny_gpt()
    model.init_weights()
    return model


class TestSparsePropLinearAutograd:
    def test_output_matches_dense_masked(self):
        """SparsePropLinear forward output matches masked nn.Linear."""
        torch.manual_seed(42)
        lin = nn.Linear(6, 4, bias=True)
        # sparsity=0.0 → all-ones mask, no zeroing; we apply a custom mask below
        sparse = SparsePropLinear.from_linear(lin, sparsity=0.0)
        with torch.no_grad():
            mask = sample_mask()
            sparse._set_mask(mask)

        x = torch.randn(8, 6, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            w_eff = lin.weight * mask.to(lin.weight.dtype)
        y_ref = torch.nn.functional.linear(x, w_eff, lin.bias)

        y_sparse = sparse(x)
        assert torch.allclose(y_sparse, y_ref, atol=1e-5), (
            f"Forward mismatch: max diff = {(y_sparse - y_ref).abs().max()}"
        )

    def test_backward_matches_dense_masked(self):
        """SparsePropLinear backward gradients match dense masked reference."""
        torch.manual_seed(42)
        lin = nn.Linear(6, 4, bias=True)
        sparse = SparsePropLinear.from_linear(lin, sparsity=0.0)

        mask = sample_mask()
        sparse._set_mask(mask)

        x = torch.randn(8, 6, dtype=torch.float32)

        # Sparse forward+backward
        x_sparse = x.clone().requires_grad_(True)
        y = sparse(x_sparse)
        loss = y.sum()
        loss.backward()

        # Dense masked reference using a separate nn.Linear with masked weight
        lin_ref = nn.Linear(6, 4, bias=True)
        with torch.no_grad():
            lin_ref.weight.copy_(lin.weight * mask.to(lin.weight.dtype))
            lin_ref.bias.copy_(lin.bias)
        x_ref = x.clone().requires_grad_(True)
        y_ref = lin_ref(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # grad_x should match
        assert torch.allclose(x_sparse.grad, x_ref.grad, atol=1e-3), (
            f"grad_x mismatch: max diff = {(x_sparse.grad - x_ref.grad).abs().max()}"
        )
        # grad_w at nnz positions should match
        assert torch.allclose(
            sparse.weight.grad[mask], lin_ref.weight.grad[mask], atol=1e-3
        ), (
            f"grad_w (nnz) mismatch: max diff = {(sparse.weight.grad[mask] - lin_ref.weight.grad[mask]).abs().max()}"
        )
        # grad_w at masked positions should be zero
        assert (sparse.weight.grad[~mask] == 0).all(), (
            "Non-zero grad at masked positions!"
        )
        # grad_bias should match
        assert torch.allclose(sparse.bias.grad, lin_ref.bias.grad, atol=1e-5), (
            f"grad_b mismatch: max diff = {(sparse.bias.grad - lin_ref.bias.grad).abs().max()}"
        )

    def test_masked_weights_stay_zero_after_step(self):
        """Optimizer step does not activate pruned weight positions."""
        torch.manual_seed(42)
        lin = nn.Linear(6, 4, bias=True)
        sparse = SparsePropLinear.from_linear(lin, sparsity=0.0)

        with torch.no_grad():
            mask = torch.rand(4, 6) < 0.5
            sparse._set_mask(mask)

        x = torch.randn(8, 6, dtype=torch.float32)
        y = sparse(x)
        loss = y.sum()
        loss.backward()

        opt = torch.optim.AdamW(sparse.parameters(), lr=0.01)
        opt.step()

        # After optimizer step, masked positions must remain zero
        assert (sparse.weight.data[~mask] == 0).all(), (
            "Pruned weights became non-zero after optimizer step"
        )

    def test_sparsity_level_correct(self):
        """Sparsity mask achieves the requested prune ratio."""
        torch.manual_seed(42)
        sparse = SparsePropLinear(100, 100, sparsity=0.6, bias=False)
        apply_static_sparsity_mask(sparse, sparsity=0.6)
        nnz = sparse.sparsity_mask.sum().item()
        total = 100 * 100
        actual_sparsity = 1.0 - nnz / total
        assert 0.55 <= actual_sparsity <= 0.65, (
            f"Sparsity {actual_sparsity:.2f} outside expected range for 0.6 target"
        )


class TestInjection:
    def test_inject_replaces_linear(self, tiny_model):
        """inject_sparseprop_layers replaces nn.Linear with SparsePropLinear."""
        from nanochat.models.quant.linear import LCQATLinear

        n_before = sum(
            1
            for m in tiny_model.modules()
            if isinstance(m, nn.Linear)
            and not isinstance(m, (SparsePropLinear, LCQATLinear))
        )
        inject_sparseprop_layers(tiny_model, sparsity=0.5)
        n_after = sum(
            1 for m in tiny_model.modules() if isinstance(m, SparsePropLinear)
        )
        assert n_after > 0, "No SparsePropLinear modules created"
        assert n_after == n_before, f"Expected {n_before} replacements, got {n_after}"

    def test_inject_preserves_structure(self, tiny_model):
        """Injected model produces same output shape."""
        x = torch.randint(0, 64, (2, 16))
        inject_sparseprop_layers(tiny_model, sparsity=0.5)
        with torch.no_grad():
            out = tiny_model(x)
        assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"

    def test_apply_static_sparsity_mask(self, tiny_model):
        """apply_static_sparsity_mask prunes every SparsePropLinear to the target."""
        inject_sparseprop_layers(tiny_model, sparsity=0.1)
        achieved = apply_static_sparsity_mask(tiny_model, sparsity=0.6)
        layers = [m for m in tiny_model.modules() if isinstance(m, SparsePropLinear)]
        assert layers, "no SparsePropLinear modules to mask"
        for m in layers:
            nnz = m.sparsity_mask.sum().item()
            total = m.weight.numel()
            actual = 1.0 - nnz / total
            assert 0.55 <= actual <= 0.65, (
                f"Module sparsity {actual:.2f} outside range for 0.6 target"
            )
            # The pruning criterion writes exact zeros, which is what makes a
            # pruned position a structural zero under the LC-QAT codebook.
            assert (m.weight.detach()[~m.sparsity_mask] == 0.0).all()
        total_all = sum(m.weight.numel() for m in layers)
        nnz_all = sum(int(m.sparsity_mask.sum()) for m in layers)
        assert achieved == pytest.approx(1.0 - nnz_all / total_all)


class TestDBCPUIntegration:
    def test_sparseprop_is_linear_subclass(self):
        """SparsePropLinear is recognized as nn.Linear by DB-CPU partitioner."""
        from nanochat.models.backbone import Linear

        sparse = SparsePropLinear(32, 32, sparsity=0.5)
        assert isinstance(sparse, nn.Linear)
        assert isinstance(sparse, Linear)

    def test_blockwise_training_runs(self, tiny_model):
        """DB-CPU block-wise training step works with SparseProp layer."""
        from nanochat.training.diffusion_blocks import (
            DiffusionBlockEngine,
            EquiProbabilityPartitioner,
        )

        inject_sparseprop_layers(tiny_model, sparsity=0.5)

        engine = DiffusionBlockEngine(
            model=tiny_model,
            partitioner=EquiProbabilityPartitioner(num_blocks=2),
        )

        idx = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))

        engine.zero_grad()
        loss = engine.train_step(idx, targets)
        loss.backward()

        # Check gradients exist on sparse weights
        has_grad = False
        for name, p in engine.named_parameters():
            if "sparsity" not in name and p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found on model parameters"

    def test_wrapping_lcqat_preserves_learned_activation_lut(self, tiny_model):
        """SparseProp must not drop the D9 learned activation tables.

        `SparsePropLinearLCQAT` re-parents the wrapped LCQATLinear's
        parameters, quantizers and buffers, then deliberately drops the
        reference to the original module. `learnable_activation_lut` is a
        *submodule* (it holds the trainable `logits` / `initial_table`
        parameters), so it was silently lost at wrap time. The checkpoint then
        saved without those keys and `build_model`'s strict load failed with
        "unexpected key(s)" on every wrapped `c_fc` -- a model trained with the
        always-on defaults could not be loaded for evaluation at all.
        """
        from nanochat.models.quant import LayerKConfig, retrofit_model
        from nanochat.models.quant.export import attach_learnable_activation_luts

        model = retrofit_model(tiny_model, LayerKConfig(min_linear_dim=1))
        n_attached = attach_learnable_activation_luts(model)
        assert n_attached > 0, "precondition: a learned table must be attached"

        before = {k for k in model.state_dict() if "learnable_activation_lut" in k}
        assert before, "precondition: the table must appear in the state_dict"

        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        after = {k for k in model.state_dict() if "learnable_activation_lut" in k}
        assert after == before, (
            "wrapping in SparseProp dropped the learned activation table: "
            f"before={sorted(before)} after={sorted(after)}"
        )

    def test_sparseprop_with_lcqat(self, tiny_model):
        from nanochat.models.quant import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        # Inject SparseProp wrapping LCQATLinear while preserving codebooks
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        # After inject with_lcqat=True, top-level LCQATLinear modules are wrapped
        # in SparsePropLinearLCQAT (which absorbs the LCQATLinear's weight/
        # quantizers in-tree, so no orphan LCQATLinear remains).
        sparse_count = sum(
            1 for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)
        )
        assert sparse_count > 0, (
            "LCQAT retrofit should produce SparsePropLinearLCQAT modules"
        )
        # No orphan LCQATLinear modules should remain after wrapping
        orphans = sum(1 for m in model.modules() if isinstance(m, LCQATLinear))
        assert orphans == 0, f"Expected no orphan LCQATLinear modules, got {orphans}"
        # Verify codebooks are preserved on the wrappers
        for m in model.modules():
            if isinstance(m, SparsePropLinearLCQAT):
                assert m.weight_quantizer is not None
                assert m.act_quantizer is not None

    def test_engine_apply_sparseprop(self, tiny_model):
        """DiffusionBlockEngine.apply_sparseprop injects SparseProp into all blocks."""
        from nanochat.training.diffusion_blocks import (
            DiffusionBlockEngine,
            EquiProbabilityPartitioner,
        )

        engine = DiffusionBlockEngine(
            model=tiny_model,
            partitioner=EquiProbabilityPartitioner(num_blocks=2),
        )
        n = engine.apply_sparseprop(sparsity=0.5)
        assert n > 1, f"Expected multiple SparsePropLinear modules, got {n}"
        # Verify engine still functional: train_step after sparseprop
        idx = torch.randint(0, 64, (2, 8))
        targets = torch.randint(0, 64, (2, 8))
        engine.zero_grad()
        loss = engine.train_step(idx, targets)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for _, p in engine.named_parameters()
        )
        assert has_grad, "No gradients after SparseProp-injected train_step"


class TestSparsePropLCQAT:
    def test_lcqat_backward_gradients_flow(self, tiny_model):
        """SparsePropLinearLCQAT backward produces gradients on all params."""
        from nanochat.models.quant import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        model.train()

        x = torch.randn(4, 8, 32, dtype=torch.float32)
        module = next(
            m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)
        )
        y = module(x)
        loss = y.sum()
        loss.backward()

        # Weight, bias, and codebook params should all have gradients
        assert module.weight.grad is not None, "weight grads missing"
        if module.bias is not None:
            assert module.bias.grad is not None, "bias grads missing"
        # Codebook params (from weight_quantizer)
        for name, p in module.weight_quantizer.named_parameters():
            assert p.grad is not None, f"Codebook param {name} has no gradient"

    def test_lcqat_sparsity_mask_stays_fixed(self, tiny_model):
        """SparsePropLinearLCQAT sparsity mask does not change during training."""
        from nanochat.models.quant import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        model.train()
        module = next(
            m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)
        )
        mask_before = module.sparsity_mask.clone()

        x = torch.randn(4, 8, 32, dtype=torch.float32)
        y = module(x)
        loss = y.sum()
        loss.backward()

        # Optimizer step
        opt = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad], lr=0.01
        )
        opt.step()

        # Mask should not change after optimizer step
        assert torch.equal(module.sparsity_mask, mask_before), (
            "Sparsity mask changed during training"
        )


class TestSparsePropDenseGEMMPath:
    """SparseProp's training path is a masked dense GEMM, not an nnz walk.

    These guard the two defects that made the sparse stack 4.27x slower than
    dense while producing no metric benefit, and the correctness bug that
    shipped alongside them. Both were invisible to the parity tests: the AVX2
    kernels still agree with the dense reference, they simply were not the code
    running during training.
    """

    @pytest.fixture
    def sparse_layer(self, tiny_model):
        from nanochat.models.quant import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.75, with_lcqat=True)
        return next(m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT))

    def test_graph_has_no_index_backward(self, sparse_layer):
        """The weight gather must not record an IndexBackward0.

        `codebook[indices]` is differentiable, so autograd adds an IndexBackward
        whose backward scatters the output gradient over the whole dense weight
        with index_put. Profiling measured that at 486 ms -- over half the step.
        """
        x = torch.randn(4, 8, sparse_layer.in_features)
        nodes: set[str] = set()

        def walk(gf):
            if gf is None or type(gf).__name__ in nodes:
                return
            nodes.add(type(gf).__name__)
            for nxt, _ in gf.next_functions:
                walk(nxt)

        walk(sparse_layer(x).grad_fn)
        assert "IndexBackward0" not in nodes, (
            "SparseProp weight gather reintroduced an IndexBackward0; route the "
            "codebook through _CodebookSTE (see SparsePropLinearLCQAT._quantize)."
        )

    def test_weight_gradient_scales_by_sqrt_keep_fraction(self, tiny_model):
        """A sparse layer's weight gradient must be sqrt(keep)x the dense one.

        `SparsePropLinearLCQAT` re-parents the quantizers but does not inherit
        `LCQATLinear._quantize`, so calling the codebook module directly used to
        skip the PRD 2.4 `inv_sqrt_n` scale, training the codebook sqrt(numel)
        times too hot -- a silent change to optimization, not a visible failure.

        The invariant that catches that is the *weight* gradient ratio. Pruning
        to `s` sparsity keeps `(1 - s)` of the entries and zeroes the rest, so
        dW masked onto the pattern carries sqrt(1 - s) of the dense gradient
        norm. Measured against the un-sparsified twin on identical inputs:

            sparsity 0.25 -> 0.862  (sqrt(keep) 0.866)
            sparsity 0.50 -> 0.704  (sqrt(keep) 0.707)
            sparsity 0.75 -> 0.502  (sqrt(keep) 0.500)

        A missing inv_sqrt_n scale instead multiplies the codebook gradient by
        sqrt(numel), which for any real layer is orders of magnitude outside this
        band -- so this single ratio pins both the mask arithmetic and the
        gradient scale.
        """
        from nanochat.models.quant import LayerKConfig, retrofit_model

        # Both arms must be the *same* layer: inject_sparseprop_layers wraps a
        # subset of the Linears, so picking "the first LCQATLinear" and "the
        # first SparsePropLinearLCQAT" independently can land on different
        # layers and compare two unrelated matrices.
        # Both arms must be the *same layer* on *independent models*.
        # `retrofit_model` mutates the model in place and re-parents its
        # quantizers, so retrofitting the same GPT twice leaves the second call
        # wrapping modules the first already replaced -- and both arms then share
        # one weight tensor, which makes the ratio exactly 1.0 and the test
        # silently vacuous.
        dense = retrofit_model(_fresh_tiny_gpt(), LayerKConfig(min_linear_dim=1))
        dense_named = {
            n: m for n, m in dense.named_modules() if isinstance(m, LCQATLinear)
        }
        sparse_model = retrofit_model(_fresh_tiny_gpt(), LayerKConfig(min_linear_dim=1))
        inject_sparseprop_layers(sparse_model, sparsity=0.75, with_lcqat=True)
        sparse_named = {
            n: m
            for n, m in sparse_model.named_modules()
            if isinstance(m, SparsePropLinearLCQAT)
        }
        shared = sorted(set(dense_named) & set(sparse_named))
        assert shared, "no layer was sparsified; test cannot run"
        name = shared[0]
        dense_layer, sparse_layer = dense_named[name], sparse_named[name]
        assert sparse_layer.weight.shape == dense_layer.weight.shape

        # Same input to both arms, so the ratio isolates the gradient path.
        x = torch.randn(4, 8, sparse_layer.in_features)
        sparse_layer(x).sum().backward()
        dense_layer(x.clone()).sum().backward()

        sparse_norm = sparse_layer.weight.grad.norm().item()
        dense_norm = dense_layer.weight.grad.norm().item()
        assert dense_norm > 0, "dense twin produced no gradient; test is vacuous"
        ratio = sparse_norm / dense_norm
        assert 0.40 < ratio < 0.60, (
            f"weight gradient is {ratio:.3f}x the dense reference; expected "
            f"~0.5 = sqrt(1 - sparsity). A much larger ratio means the "
            f"inv_sqrt_n gradient scale is not being applied."
        )

    def test_pruned_slots_hold_exact_zeros(self, sparse_layer):
        """Structural zeros must stay == 0.0, not merely near-zero.

        The dense GEMM relies on this: it is what makes the masked product sum
        exactly the surviving SpMM terms, and it is the contract the sparse
        export and mul-less kernels read.
        """
        x = torch.randn(4, 8, sparse_layer.in_features)
        sparse_layer(x).sum().backward()
        mask = sparse_layer.sparsity_mask
        assert torch.equal(
            sparse_layer.weight.detach()[~mask],
            torch.zeros_like(sparse_layer.weight.detach()[~mask]),
        ), "pruned slots lost their exact zero anchor"

    def test_weight_grad_is_zero_on_pruned_slots(self, sparse_layer):
        """The masked backward must not accumulate gradient into pruned slots."""
        x = torch.randn(4, 8, sparse_layer.in_features)
        sparse_layer(x).sum().backward()
        mask = sparse_layer.sparsity_mask
        assert torch.equal(
            sparse_layer.weight.grad[~mask],
            torch.zeros_like(sparse_layer.weight.grad[~mask]),
        )

    def test_output_values_lie_on_the_out_codebook(self, tiny_model):
        """The layer's output must be snapped onto its out_quantizer's levels.

        `__init__` re-parents `out_quantizer` off the wrapped LCQATLinear -- it
        has to, or the c_q/c_k/c_v and c_fc layers silently lose KV-cache
        quantization. But the forward never called it, so every layer carrying an
        out_quantizer trained with an unquantized output while its dense twin did
        not. That made the SparseProp arm measure ~30-47% "faster" end to end,
        purely because it was skipping a quantization pass.

        The check is on the *values*, not on a re-run: quantizing an already
        quantized output is trivially a no-op for a good implementation and also
        for a missing one, so idempotence proves nothing here. Only the forward
        snapping onto the 15 codebook levels distinguishes them.
        """
        from nanochat.models.quant import LayerKConfig, retrofit_model

        model = retrofit_model(_fresh_tiny_gpt(), LayerKConfig(min_linear_dim=1))
        inject_sparseprop_layers(model, sparsity=0.75, with_lcqat=True)
        layer = next(
            m
            for m in model.modules()
            if isinstance(m, SparsePropLinearLCQAT)
            and getattr(m, "out_quantizer", None) is not None
        )
        out = layer(torch.randn(4, 8, layer.in_features))
        levels = layer.out_quantizer.get_codebook().reshape(-1)
        # Every output value must equal some codebook level exactly.
        distances = (out.reshape(-1, 1) - levels.reshape(1, -1)).abs().min(dim=1).values
        assert torch.equal(distances, torch.zeros_like(distances)), (
            f"{int((distances > 0).sum())} of {distances.numel()} outputs are not "
            "on a codebook level; out_quantizer is not being applied"
        )

    def test_out_quantizer_receives_gradient(self, tiny_model):
        """The re-parented out_quantizer must be trainable, not dead weight."""
        from nanochat.models.quant import LayerKConfig, retrofit_model

        torch.manual_seed(0)
        model = retrofit_model(tiny_model, LayerKConfig(min_linear_dim=1))
        inject_sparseprop_layers(model, sparsity=0.75, with_lcqat=True)
        layer = next(
            m
            for m in model.modules()
            if isinstance(m, SparsePropLinearLCQAT)
            and getattr(m, "out_quantizer", None) is not None
        )
        layer(torch.randn(4, 8, layer.in_features)).sum().backward()
        grads = [
            p.grad
            for n, p in layer.named_parameters()
            if "out_quantizer" in n and p.requires_grad
        ]
        assert grads, "out_quantizer exposes no trainable parameters"
        assert any(g is not None and bool((g != 0).any()) for g in grads), (
            "out_quantizer received no gradient; it is being carried but not used"
        )


def _fresh_tiny_gpt() -> nn.Module:
    """A newly initialized tiny GPT.

    Every arm of a comparison needs its own instance: `retrofit_model` and
    `inject_sparseprop_layers` both mutate in place and re-parent modules, so
    reusing one model across arms leaves the arms sharing weight tensors and
    makes any gradient comparison trivially equal.
    """
    return build_tiny_gpt()


class TestSparsePropLearnedActivationLUT:
    """The sparse wrapper must reproduce the dense layer's activation path.

    `SparsePropLinearLCQAT.apply_trained_activation` re-implemented the gather as
    `lut.resolved_table()[indices]` instead of calling the table's forward. That
    is ~3x cheaper (14 ms vs 45 ms at [4,512,1536]) because it drops the
    straight-through relaxation: `LearnableIndexLut.forward` returns
    `soft + hard - soft.detach()`, so the forward value is unchanged but the
    gradient reaches the trained parameters, while the manual gather returns
    `hard` alone. Every SparseProp run therefore trained `c_fc` through a dead
    LUT, silently disabling --lcqat-lut-relaxation and --lcqat-act-body.
    """

    @pytest.fixture
    def models(self):
        from nanochat.models.quant import LayerKConfig, retrofit_model
        from nanochat.models.quant.export import attach_learnable_activation_luts

        def build():
            model = build_tiny_gpt()
            retrofit_model(model, LayerKConfig(min_linear_dim=1))
            n = attach_learnable_activation_luts(
                model, relaxation="logits", act_body="pwl"
            )
            assert n, "no activation LUT attached; test cannot run"
            return model

        torch.manual_seed(0)
        dense = build()
        torch.manual_seed(0)
        sparse = build()
        # Share every weight so the two arms differ only in the activation path.
        # `retrofit_model` re-parents quantizers, so copy by module name, not by
        # parameter position.
        dense_sd = dense.state_dict()
        with torch.no_grad():
            for name, param in sparse.named_parameters():
                if name in dense_sd:
                    param.copy_(dense_sd[name])
        inject_sparseprop_layers(sparse, sparsity=0.75, with_lcqat=True)
        return dense, sparse

    def test_lut_survives_injection(self, models):
        """Injection must not drop the trained table off c_fc."""
        _, sparse = models
        wrapped = [
            m
            for m in sparse.modules()
            if isinstance(m, SparsePropLinearLCQAT)
            and getattr(m, "learnable_activation_lut", None) is not None
        ]
        assert wrapped, "inject_sparseprop_layers dropped learnable_activation_lut"

    def test_activation_matches_dense_twin(self, models):
        """Sparse and dense c_fc must emit the same activation values."""
        dense, sparse = models
        d_fc = dense.transformer.h[0].mlp.c_fc
        s_fc = sparse.transformer.h[0].mlp.c_fc
        assert s_fc.in_features == d_fc.in_features

        # Perturb the trained table off its init so the relaxation is observable:
        # at init the hard and soft paths agree, and a dead LUT would pass.
        # The SAME perturbation is applied to both arms -- a fresh draw per model
        # would make them differ and mask the regression this test exists to catch.
        torch.manual_seed(1234)
        for model in (dense, sparse):
            for mod in model.modules():
                lut = getattr(mod, "learnable_activation_lut", None)
                if (
                    lut is not None
                    and hasattr(lut, "logits")
                    and lut.logits is not None
                ):
                    with torch.no_grad():
                        lut.logits.add_(torch.randn_like(lut.logits) * 0.1)

        y = torch.randn(4, 8, d_fc.out_features)
        d_out = d_fc.apply_trained_activation(y)
        s_out = s_fc.apply_trained_activation(y)
        assert s_out.shape == d_out.shape
        # float32 tolerance, not bit-equality: the two arms reach the same values
        # through a different op order (the sparse layer's dense GEMM vs the
        # dense layer's F.linear), which differs by at most an ULP. Before the
        # fix this differed by the relaxation term itself, not by rounding --
        # see the gradient test for the sharp version of the invariant.
        torch.testing.assert_close(s_out, d_out, rtol=1e-6, atol=1e-6)

    def test_lut_parameters_receive_gradient(self, models):
        """The trained table must be live, not carried-but-unused."""
        _, sparse = models
        layer = next(
            m
            for m in sparse.modules()
            if isinstance(m, SparsePropLinearLCQAT)
            and getattr(m, "learnable_activation_lut", None) is not None
        )
        y = torch.randn(4, 8, layer.out_features)
        layer.apply_trained_activation(y).sum().backward()
        lut = layer.learnable_activation_lut
        trained = [
            p
            for p in lut.parameters()
            if p.requires_grad and p.grad is not None and bool((p.grad != 0).any())
        ]
        assert trained, (
            "learnable_activation_lut received no gradient; the LUT is dead on "
            "the SparseProp path"
        )
