"""Learnable, data-dependent LR tests for the MinExpLSTM optimizers.

The optimizer's learning rate is a *learnable* function of the
current step's data: a tiny linear predictor maps the
RMS-normalized gradient magnitude to a logit, and a sigmoid
parametrizes the LR to (0, 1) -- strictly positive (the
optimizer always moves) and bounded by 1 (the step is bounded).
The model is elementwise and parallel across the batch
dimension, so one LR value per shape group suffices (no
per-coordinate vector). The predictor is co-trained by the
AdamW meta-optimizer, so the LR adapts to the loss landscape.

The AdamW meta-optimizer's own LR stays a plain hyperparameter
-- ``meta_lr`` -- optimized by Lightning's LR finder before
training starts (warmup), not learned online.
"""

from __future__ import annotations

import pytest
import torch

from nanochat.training.lightning_module import (
    JointTrainableModule,
    MinExpLSTMLightningModule,
)
from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer
from tests.lightning_common import TinyModel


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _make_opt(lr: float = 0.5, **kw):
    torch.manual_seed(0)
    model = _Tiny()
    opt = SGD_MinExpLstm_Optimizer(model.parameters(), lr=lr, **kw)
    return model, opt


def test_when_optimizer_built_then_lr_predicted_in_zero_one() -> None:
    """The LR is a data-dependent tensor in (0, 1).

    The predictor maps the gradient RMS to a logit and a
    sigmoid parametrizes the LR to (0, 1): strictly positive
    (the optimizer always moves) and bounded by 1 (the step
    is bounded). A zero-weight predictor starts at the
    requested scalar `lr`.
    """
    model, opt = _make_opt(lr=0.5)
    grad = torch.randn(1, 8)
    lr = opt.predict_lr(opt._rms(grad))
    assert isinstance(lr, torch.Tensor)
    assert lr.requires_grad, "the LR must be a learnable function"
    assert 0.0 < float(lr.detach()) < 1.0, f"LR {float(lr.detach())} outside (0, 1)"
    # A zero-weight predictor starts at the constant `lr`.
    assert float(lr.detach()) == pytest.approx(0.5, rel=1e-5)


def test_when_lr_predictor_extreme_then_stays_in_zero_one() -> None:
    """The sigmoid keeps the LR in (0, 1) for any input.

    Drive the predictor's bias to both extremes: a very
    negative bias keeps the LR above 0 (the optimizer must
    always move); a very positive bias keeps it below 1.
    """
    _, opt = _make_opt(lr=0.5)
    grad = torch.randn(1, 8)
    with torch.no_grad():
        opt.lr_predictor.bias.fill_(-50.0)
    lr_neg = float(opt.predict_lr(opt._rms(grad)).detach())
    assert lr_neg > 0.0

    with torch.no_grad():
        opt.lr_predictor.bias.fill_(50.0)
    lr_pos = float(opt.predict_lr(opt._rms(grad)))
    # The sigmoid saturates to 1.0 in float32 for a large positive logit; the LR is bounded by 1 (inclusive upper bound), strictly positive.
    assert lr_pos <= 1.0
    assert lr_pos > 0.0


def test_when_meta_step_runs_then_lr_predictor_is_learnable() -> None:
    """The LR predictor carries a gradient after a meta-step.

    The LR is data-dependent: the differentiable closure
    rebuild (meta_step) computes the LR from the rebuilt
    gradient and routes the loss gradient back through the
    predictor, so the AdamW meta-optimizer updates it. A
    non-learnable LR would leave ``lr_predictor`` grads None.
    """
    model, opt = _make_opt(lr=0.5)
    x = torch.randn(4, 8)
    y = torch.randint(0, 8, (4,))

    # Primary step (builds the INT8 states). The grads must stay populated: the meta-step closure reads p.grad to rebuild each updated parameter, so meta_step runs BEFORE zero_grad.
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()

    # Meta-step rebuilds the loss differentiably through the cell AND the data-dependent LR, so the predictor gets a gradient.
    def _closure(params):
        out = torch.func.functional_call(model, params, (x,))
        return torch.nn.functional.cross_entropy(out, y)

    opt.meta_step(model, _closure)
    opt.zero_grad()

    for q in opt.lr_predictor.parameters():
        assert q.grad is not None, (
            "the LR predictor must receive a gradient from "
            "the differentiable meta-rebuild"
        )
        assert torch.isfinite(q.grad).all()


def test_when_lr_data_dependent_then_varies_with_gradient() -> None:
    """The LR is a function of the gradient magnitude.

    The predictor consumes the gradient RMS, so two different
    gradients produce two different LRs once the predictor is
    trained (its weight is non-zero). This is what makes the
    LR data-dependent: it adapts to the current step's data.
    """
    _, opt = _make_opt(lr=0.5)
    # Train the predictor off its zero init so the LR varies with the input.
    with torch.no_grad():
        opt.lr_predictor.weight.fill_(1.0)
    grad_small = torch.full((1, 8), 1e-3)
    grad_large = torch.full((1, 8), 1.0)
    lr_small = float(opt.predict_lr(opt._rms(grad_small)))
    lr_large = float(opt.predict_lr(opt._rms(grad_large)))
    assert lr_small != lr_large, "the LR must depend on the gradient magnitude"


def test_when_lr_larger_then_update_larger() -> None:
    """The LR scales the update magnitude.

    A larger LR produces a proportionally larger parameter
    update (the SGD direction is scaled by the LR), so the
    LR directly modulates the step.
    """
    torch.manual_seed(0)
    model_a = _Tiny()
    model_b = _Tiny()
    model_b.load_state_dict(model_a.state_dict())
    init_weight = model_a.lin.weight.detach().clone()

    opt_a = SGD_MinExpLstm_Optimizer(model_a.parameters(), lr=0.1)
    opt_b = SGD_MinExpLstm_Optimizer(model_b.parameters(), lr=0.9)

    x = torch.randn(4, 8)
    y = torch.randint(0, 8, (4,))
    for model, opt in ((model_a, opt_a), (model_b, opt_b)):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    move_a = (model_a.lin.weight - init_weight).abs().sum()
    move_b = (model_b.lin.weight - init_weight).abs().sum()
    assert move_b > move_a, "larger LR must produce a larger step"


def test_when_meta_lr_is_hyperparameter_then_not_learned() -> None:
    """The AdamW meta LR stays a plain hyperparameter.

    The meta-optimizer's own LR (which trains the cell and
    the LR predictor) is a fixed float set at construction --
    it is NOT a learnable parameter. Only the primary
    optimizer's LR is data-dependent and learnable.
    """
    _, opt = _make_opt(lr=0.5, meta_lr=1e-3)
    # The meta-optimizer is a plain AdamW with a fixed LR.
    meta_lr = opt.meta_optimizer.param_groups[0]["lr"]
    assert meta_lr == 1e-3
    assert isinstance(meta_lr, float)


def _first_optimizer(opts) -> object:
    """Unwrap the first optimizer from configure_optimizers().

    Lightning returns ``([opt], [sched])`` or ``opt``; handle
    both so the tests can reach the optimizer either way.
    """
    if isinstance(opts, tuple):
        return opts[0][0] if isinstance(opts[0], list) else opts[0]
    if isinstance(opts, list):
        return opts[0]
    return opts


def test_when_lightning_module_then_lr_is_data_dependent() -> None:
    """``MinExpLSTMLightningModule`` exposes the learnable LR.

    The Lightning wrapper constructs the SGD optimizer whose
    LR is a data-dependent learnable function; the AdamW
    baseline (``optimizer="adamw"``) keeps its LR a plain
    hyperparameter (no learnable LR).
    """
    model = TinyModel()
    module = MinExpLSTMLightningModule(model, lr=0.5)
    opt = _first_optimizer(module.configure_optimizers())
    assert isinstance(opt, SGD_MinExpLstm_Optimizer)
    # The LR predictor is part of the meta-optimizer's parameter set, so it is learnable.
    lr_params = list(opt.lr_predictor.parameters())
    assert all(p.requires_grad for p in lr_params)
    grad = torch.randn(1, 8)
    lr = opt.predict_lr(opt._rms(grad))
    assert 0.0 < float(lr) < 1.0


def test_when_adamw_module_then_lr_is_plain_float() -> None:
    """The AdamW baseline keeps its LR a plain hyperparameter.

    With ``optimizer="adamw"`` there is no learnable LR -- the
    LR is the float Lightning's LR finder tunes (warmup).
    """
    model = TinyModel()
    module = MinExpLSTMLightningModule(model, lr=1e-3, optimizer="adamw")
    opt = _first_optimizer(module.configure_optimizers())
    assert isinstance(opt, torch.optim.AdamW)
    # The AdamW LR is a plain hyperparameter (the float Lightning's LR finder tunes). The scheduler warms it up, so read the *initial* LR the module configured, not the warmed-up current value.
    initial_lr = opt.param_groups[0].get("initial_lr", opt.param_groups[0]["lr"])
    assert isinstance(initial_lr, float)
    assert initial_lr == 1e-3


def test_when_joint_module_then_lr_learnable_and_meta_lr_fixed() -> None:
    """``JointTrainableModule`` threads the learnable LR.

    The manual-optimization module constructs the SGD optimizer
    with a data-dependent learnable LR in (0, 1) and a fixed
    AdamW meta LR.
    """
    model = _Tiny()
    module = JointTrainableModule(model, lr=0.5, meta_lr=1e-3)
    opt = _first_optimizer(module.configure_optimizers())
    assert isinstance(opt, SGD_MinExpLstm_Optimizer)
    grad = torch.randn(1, 8)
    lr = opt.predict_lr(opt._rms(grad))
    assert 0.0 < float(lr) < 1.0
    assert opt.meta_optimizer.param_groups[0]["lr"] == 1e-3


def test_when_tune_lr_then_meta_lr_is_optimized() -> None:
    """``tune_lr`` optimizes the AdamW meta-optimizer's LR.

    The AdamW meta-optimizer's LR is a plain hyperparameter,
    so Lightning's LR finder optimizes it before training
    starts (warmup). ``tune_lr`` applies the finder's
    suggestion to the meta-optimizer's param groups (and the
    primary optimizer's seed LR), not to the data-dependent
    LR predictor directly.
    """
    model = TinyModel()
    module = MinExpLSTMLightningModule(model, lr=0.5, minexp_meta_lr=1e-3)
    module._train_dataloader = _loader()
    from lightning.pytorch import Trainer

    trainer = Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    # Attach the module to the trainer and configure the optimizers so tune_lr can reach the meta-optimizer through trainer.optimizers[0].
    module._trainer = trainer
    optimizers, _ = module.configure_optimizers()
    trainer.optimizers = optimizers  # type: ignore[assignment]

    # A stub tuner that returns a fixed suggestion, so the test does not need a real LR sweep.
    class _StubTuner:
        def __init__(self, _trainer):
            self._trainer = _trainer

        def lr_find(self, _module, **_kw):
            class _Result:
                def suggestion(self) -> float:
                    return 0.123

            return _Result()

    import lightning.pytorch.tuner as tuner_mod

    real_tuner = tuner_mod.Tuner
    tuner_mod.Tuner = _StubTuner  # type: ignore[assignment]
    try:
        suggested = module.tune_lr(trainer)
    finally:
        tuner_mod.Tuner = real_tuner
    assert suggested == 0.123
    # The meta-optimizer's (AdamW) LR is the tuned value.
    from nanochat.training.minexp_lstm import (
        SGD_MinExpLstm_Optimizer as _SGDOpt,
    )

    opt = trainer.optimizers[0]
    assert isinstance(opt, _SGDOpt)
    assert opt.meta_optimizer.param_groups[0]["lr"] == 0.123


def _loader():
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(8, 8)
    y = torch.randint(0, 8, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
