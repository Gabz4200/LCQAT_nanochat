"""Lightning integration tests for ``JointTrainableModule`` (SGD-MinExpLSTM).

The SGD twin of ``tests/test_mezo_lightning.py``. These tests pin
the joint-training contract that previously had zero coverage: the
primary FiLM-modulated step, the online meta-optimization pass
(cell co-training through a differentiable closure), and the
meta-step/zero_grad ordering the closure depends on.
"""

from __future__ import annotations

import pytest
import torch

from nanochat.training.lightning_module import (
    JointTrainableModule,
    MeZoJointTrainableModule,
)
from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer
from tests.lightning_common import TinyModel, dataloader, trainer


def _inner_optimizer(module) -> SGD_MinExpLstm_Optimizer:
    """The module's single SGD_MinExpLstm_Optimizer.

    After ``fit`` the Trainer adopts ``configure_optimizers``'
    optimizer, so read it from the strategy (the trained
    instance, not a freshly built one). Before ``fit`` fall
    back to ``configure_optimizers`` directly.
    """
    try:
        t = module.trainer
    except (RuntimeError, AttributeError):
        t = None
    strategy = getattr(t, "strategy", None) if t is not None else None
    if strategy is not None and getattr(strategy, "optimizers", None):
        opt = strategy.optimizers[0]
    else:
        opt = module.configure_optimizers()
        opt = opt[0] if isinstance(opt, list) else opt
    assert isinstance(opt, SGD_MinExpLstm_Optimizer)
    return opt


def test_when_joint_module_trains_then_loss_finite() -> None:
    """A few Lightning steps run end-to-end with finite loss."""
    module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=2)
    t = trainer(max_steps=4)
    t.fit(module, dataloader())
    # The training loop completed and logged a finite CE loss.
    # A NaN/Inf loss would mean the FiLM update diverged, so the
    # assertion is about the logged training loss, not just that
    # fit returned.
    assert module.trainer is not None
    logged = module.trainer.callback_metrics.get("train_loss")
    assert logged is not None, "JointTrainableModule must log train_loss"
    assert torch.isfinite(logged).all(), f"train_loss not finite: {logged}"


def test_when_meta_interval_hits_then_cell_trains() -> None:
    """The online meta-pass co-trains the cell through the closure.

    The closure rebuilds each updated parameter from its gradient
    and backprops through the cell's (gamma, delta), so the cell
    parameters must move once the meta-interval fires.
    """
    module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=1)
    # The Trainer adopts configure_optimizers' optimizer, so
    # read the cell after fit (the module's own optimizer,
    # not a freshly built one).
    t = trainer(max_steps=4)
    t.fit(module, dataloader())
    opt = _inner_optimizer(module)
    # After fit the cell has been through the closure
    # meta-pass (meta_interval=1 fires every step). Its
    # parameters differ from their zero-initialized film
    # head only if the closure trained them -- check the
    # film head moved off zero.
    film_bias = opt.cell.film_proj.bias.detach()
    assert float(film_bias.abs().sum()) > 0.0, (
        "the closure meta-pass must move the FiLM head off its zero init"
    )


def test_when_meta_step_before_zero_grad_then_no_stray_grads() -> None:
    """The closure reads p.grad, so meta_step runs before zero_grad.

    The closure's backward populates the model gradients; the
    module clears them after meta_step so they do not accumulate
    onto the next step's manual_backward.
    """
    module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=1)
    t = trainer(max_steps=4)
    t.fit(module, dataloader())
    model_grads = [p.grad for p in module.target_model.parameters()]
    assert all(g is None for g in model_grads), (
        "model gradients must be cleared after each training step"
    )


def test_when_meta_step_zeroed_grads_then_raises() -> None:
    """meta_step with zeroed gradients fails loudly (no silent no-op).

    The closure rebuilds every optimizer-owned parameter from
    ``p.grad``. A zeroed gradient leaves nothing to rebuild; the
    historical behavior was a silent no-op that left the cell
    untrained while reporting a healthy loss. Fail fast instead.
    """
    module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=2)
    opt = _inner_optimizer(module)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    loss = module.loss_fn(module(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()  # the historical bug: grads gone before meta_step

    def closure(params):
        out = torch.func.functional_call(module.target_model, params, (x,))
        return module.loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))

    with pytest.raises(RuntimeError, match="rebuilt 0 parameters"):
        opt.meta_step(module.target_model, closure)


def test_when_closure_rebuild_is_differentiable_then_cell_grads_populated() -> None:
    """The closure's backward populates the cell gradients."""
    module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=1)
    opt = _inner_optimizer(module)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    loss = module.loss_fn(module(x), y)
    loss.backward()
    opt.step()

    def closure(params):
        out = torch.func.functional_call(module.target_model, params, (x,))
        return module.loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))

    opt.meta_step(module.target_model, closure)
    cell_grads = [q.grad for q in opt.cell.parameters()]
    assert any(g is not None for g in cell_grads), (
        "the closure backward must populate cell gradients"
    )


def test_when_sgd_and_mezo_modules_run_then_state_is_independent() -> None:
    """SGD and MeZO joint modules train side by side without shared state.

    Each module needs its own Trainer: a single Trainer caches the
    optimizer from the first fit and hands it to the second module.
    """
    sgd_module = JointTrainableModule(TinyModel(), lr=1e-2, meta_interval=2)
    mezo_module = MeZoJointTrainableModule(
        TinyModel(), lr=1e-2, meta_interval=2, backend="naive"
    )
    for module in (sgd_module, mezo_module):
        t = trainer(max_steps=4)
        t.fit(module, dataloader())
    sgd_opt = _inner_optimizer(sgd_module)
    mezo_opt = mezo_module.configure_optimizers()
    mezo_opt = mezo_opt[0] if isinstance(mezo_opt, list) else mezo_opt
    mezo_inner = getattr(mezo_opt, "_optimizer", mezo_opt)
    # The two cells are distinct objects. The MeZO cell
    # carries the FIR kernel (conv_w) the SGD cell lacks,
    # so compare identity on the shared parameters, not
    # the parameter count.
    sgd_cell = list(sgd_opt.cell.parameters())
    mezo_cell = list(mezo_inner.cell.parameters())
    assert len(sgd_cell) < len(mezo_cell), (
        "MeZO cell has the FIR kernel (conv_w) the SGD cell lacks"
    )
    for s in sgd_cell:
        assert all(s is not m for m in mezo_cell), (
            "SGD and MeZO cells must be independent"
        )
    # The SGD optimizer has no MeZO-specific state.
    assert not hasattr(sgd_opt, "seed_history")
    assert not hasattr(sgd_opt, "mu_history")
