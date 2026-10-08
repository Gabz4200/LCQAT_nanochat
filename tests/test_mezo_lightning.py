"""Lightning integration tests for the MeZO-MinExpLSTM module.

Pins the ``MeZoJointTrainableModule`` (manual optimization, dual
forward passes, seed-replayed perturbation, FiLM-modulated update,
online REINFORCE meta-step) against a real Lightning ``Trainer``
run, and proves coexistence with the first-order
``JointTrainableModule`` (the SGD-MinExpLSTM stack) -- both
optimizers live in the same process, on the same model class, with
independent state.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset

from nanochat.training.lightning_module import (
    JointTrainableModule,
    MeZoJointTrainableModule,
)


class TinyModel(nn.Module):
    """Two-linear model with a nonzero-gradient CE loss."""

    def __init__(self, d: int = 8, vocab: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, vocab))

    def forward(self, x):
        return self.net(x)


def _dataloader(n: int = 16, d: int = 8, batch_size: int = 4) -> DataLoader:
    """Random CE dataloader (avoids the lightning.demos import)."""
    x = torch.randn(n, d)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_when_mezo_module_trains_then_loss_finite() -> None:
    """A few Lightning steps run end-to-end with finite loss."""
    model = TinyModel()
    module = MeZoJointTrainableModule(model, lr=1e-2, backend="naive")
    data = _dataloader(n=16)
    trainer = Trainer(
        max_steps=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, data)
    # The MeZO loss is logged every step; it must be finite
    # (a NaN/Inf loss would mean the zeroth-order estimate
    # diverged). Read the logged value, not a fresh random
    # draw -- the assertion must be about the training loss.
    logged = module.trainer.callback_metrics.get("train_loss")
    assert logged is not None, "MeZoJointTrainableModule must log train_loss"
    assert torch.isfinite(logged).all(), f"train_loss not finite: {logged}"


def test_when_mezo_module_steps_then_seed_history_rolls() -> None:
    """Each training step appends one (seed, mu) pair to the FIR history.

    The history grows by one per step until it reaches K (the FIR
    window), then rolls. Two steps with K=4 are still in the warmup
    phase, so the history holds exactly the step count -- not the
    full window -- and the seed and mu histories stay locked together.
    """
    model = TinyModel()
    module = MeZoJointTrainableModule(model, lr=1e-2, K=4, backend="naive")
    data = _dataloader(n=4)
    trainer = Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, data)
    opt = module.optimizers()
    opt = opt[0] if isinstance(opt, list) else opt
    # Two steps taken, K=4: the FIR window is not full yet, so
    # the history holds exactly the step count. A history that
    # grew by more than one entry per step (or none) would mean
    # the perturbation schedule desynchronized from the training
    # loop and the replayed FIR signal would be wrong.
    assert len(opt.seed_history) == 2, (
        f"expected 2 history entries after 2 steps, got {len(opt.seed_history)}"
    )
    assert len(opt.mu_history) == len(opt.seed_history)
    # Once the window fills, the history must not exceed K.
    assert len(opt.seed_history) <= opt.K


def test_when_both_optimizers_run_then_state_is_independent() -> None:
    """SGD-MinExpLSTM and MeZO-MinExpLSTM coexist without shared state."""
    torch.manual_seed(0)
    sgd_model = TinyModel()
    mezo_model = TinyModel()
    # Same init for a fair comparison of the two optimizers' effects.
    mezo_model.load_state_dict(sgd_model.state_dict())

    sgd_module = JointTrainableModule(sgd_model, lr=1e-2, meta_interval=1)
    mezo_module = MeZoJointTrainableModule(
        mezo_model, lr=1e-2, meta_interval=1, backend="naive"
    )

    data = _dataloader(n=4)
    # Separate trainers: a single ``Trainer`` caches the
    # optimizer from the first ``fit`` and would hand the
    # second module the first module's optimizer.
    sgd_trainer = Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    mezo_trainer = Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    sgd_trainer.fit(sgd_module, data)
    mezo_trainer.fit(mezo_module, data)

    # Unwrap Lightning's optimizer wrappers to the inner
    # optimizers: ``module.optimizers()`` returns the
    # ``LightningOptimizer`` wrapper, and both modules
    # wrap their own distinct optimizer.
    def _inner(module):
        opts = module.optimizers()
        opt = opts[0] if isinstance(opts, list) else opts
        return getattr(opt, "_optimizer", opt)

    sgd_opt = _inner(sgd_module)
    mezo_opt = _inner(mezo_module)
    assert isinstance(sgd_opt, torch.optim.Optimizer)
    assert isinstance(mezo_opt, torch.optim.Optimizer)
    # The two modules wrap *different* MeZO optimizer
    # instances (each ``configure_optimizers`` call builds
    # its own), so the optimizers do not share state.
    assert sgd_opt is not mezo_opt
    # The MeZO optimizer carries the FIR seed history the
    # SGD one lacks.
    assert hasattr(mezo_opt, "seed_history")
    assert not hasattr(sgd_opt, "seed_history")


def test_when_mezo_module_runs_gpu_backend_then_steps() -> None:
    """The GPU (Taichi) backend runs through the Lightning module and steps."""
    model = TinyModel()
    module = MeZoJointTrainableModule(model, lr=1e-2, backend="gpu")
    data = _dataloader(n=2)
    trainer = Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    weights_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    try:
        trainer.fit(module, data)
    except Exception as e:
        if "Vulkan" in str(e):
            pytest.skip(f"GPU backend not available: {e}")
        raise
    # The GPU backend must actually step the model: a kernel
    # that ran but left every weight unchanged would pass a
    # smoke test while silently no-opping the optimizer.
    changed = any(
        not torch.equal(p.detach(), weights_before[n])
        for n, p in model.named_parameters()
    )
    assert changed, "gpu backend did not update any weight"
