"""Integration tests for SparsePropLinear module: autograd equivalence,
mask integrity, and DB-CPU block-wise integration."""
import pytest
import torch
import torch.nn as nn

from nanochat.lcqat.linear import LCQATLinear
from nanochat.lcqat.sparseprop import (
    SparsePropLinear,
    SparsePropLinearLCQAT,
    apply_static_sparsity_mask,
    inject_sparseprop_layers,
)


@pytest.fixture
def tiny_model():
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=32, vocab_size=64, n_layer=2, n_head=2,
        n_kv_head=2, n_embd=32, window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
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
            mask = torch.tensor(
                [[1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1],
                 [1, 1, 1, 0, 0, 1],
                 [0, 0, 1, 1, 1, 0]], dtype=torch.bool
            )
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

        mask = torch.tensor(
            [[1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1],
             [1, 1, 1, 0, 0, 1],
             [0, 0, 1, 1, 1, 0]], dtype=torch.bool
        )
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
        assert torch.allclose(sparse.weight.grad[mask], lin_ref.weight.grad[mask], atol=1e-3), (
            f"grad_w (nnz) mismatch: max diff = {(sparse.weight.grad[mask] - lin_ref.weight.grad[mask]).abs().max()}"
        )
        # grad_w at masked positions should be zero
        assert (sparse.weight.grad[~mask] == 0).all(), "Non-zero grad at masked positions!"
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
        from nanochat.lcqat.linear import LCQATLinear
        n_before = sum(
            1 for m in tiny_model.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, (SparsePropLinear, LCQATLinear))
        )
        inject_sparseprop_layers(tiny_model, sparsity=0.5)
        n_after = sum(
            1 for m in tiny_model.modules()
            if isinstance(m, SparsePropLinear)
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
        """apply_static_sparsity_mask sets masks on SparsePropLinear modules."""
        inject_sparseprop_layers(tiny_model, sparsity=0.1)
        apply_static_sparsity_mask(tiny_model, sparsity=0.6)
        for m in tiny_model.modules():
            if isinstance(m, SparsePropLinear):
                nnz = m.sparsity_mask.sum().item()
                total = m.weight.numel()
                actual = 1.0 - nnz / total
                assert 0.50 <= actual <= 0.70, (
                    f"Module sparsity {actual:.2f} outside range for 0.6 target"
                )


class TestDBCPUIntegration:
    def test_sparseprop_is_linear_subclass(self):
        """SparsePropLinear is recognized as nn.Linear by DB-CPU partitioner."""
        from nanochat.gpt import Linear
        sparse = SparsePropLinear(32, 32, sparsity=0.5)
        assert isinstance(sparse, nn.Linear)
        assert isinstance(sparse, Linear)

    def test_blockwise_training_runs(self, tiny_model):
        """DB-CPU block-wise training step works with SparseProp layer."""
        from nanochat.diffusion_blocks import (
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

    def test_sparseprop_with_lcqat(self, tiny_model):
        from nanochat.lcqat import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        # Inject SparseProp wrapping LCQATLinear while preserving codebooks
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        # After inject with_lcqat=True, top-level LCQATLinear modules are wrapped
        # in SparsePropLinearLCQAT (which absorbs the LCQATLinear's weight/
        # quantizers in-tree, so no orphan LCQATLinear remains).
        sparse_count = sum(1 for m in model.modules() if isinstance(m, SparsePropLinearLCQAT))
        assert sparse_count > 0, "LCQAT retrofit should produce SparsePropLinearLCQAT modules"
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
        from nanochat.diffusion_blocks import (
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
        from nanochat.lcqat import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        model.train()

        x = torch.randn(4, 8, 32, dtype=torch.float32)
        module = next(m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT))
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
        from nanochat.lcqat import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)

        model.train()
        module = next(m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT))
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
        assert torch.equal(module.sparsity_mask, mask_before), "Sparsity mask changed during training"
