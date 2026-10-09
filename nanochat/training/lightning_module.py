"""PyTorch Lightning wrappers for the SGD-MinExpLSTM optimizer.

Two entry points:

* ``MinExpLSTMLightningModule``: the generic wrapper. It trains
  whatever ``nn.Module`` it is handed with the SGD-MinExpLSTM
  optimizer (AdamW via ``optimizer="adamw"`` for ablations) under
  the same warmup + warmdown LR shape the custom loop uses, plus a
  ``Trainer`` factory exposing Lightning's LR tuning and
  checkpointing machinery.
* ``JointTrainableModule`` (PRD section 5.3): the joint-training
  module. Manual optimization; every ``meta_interval`` steps the
  cell co-trains through a differentiably rebuilt loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy
from torch.utils.data import DataLoader

from nanochat.training.mezo_minexplstm import MeZo_MinExpLstm_Optimizer
from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer


class MinExpLSTMLightningModule(LightningModule):
    """Lightning wrapper: model + MinExpLSTM optimizer + LR schedule.

    ``loss_fn(module, batch) -> Tensor`` is the training objective;
    it is called with the module and a batch from ``train_dataloader``.
    The default is next-token cross-entropy for a GPT-style model
    whose ``forward(idx, targets)`` returns the loss, so a bare
    ``GPT`` trains out of the box.

    The LR schedule is the same warmup + warmdown shape the custom
    loop uses (linear warmup from ``init_lr_frac``, flat, then
    linear warmdown to ``final_lr_frac``), applied as a
    ``LambdaLR`` on every param group's ``initial_lr``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        optimizer: str = "minexplstm",
        minexp_lam: float = 0.10,
        minexp_clamp: float = 1.0,
        minexp_meta_lr: float = 1e-3,
        warmup_steps: int = 40,
        warmdown_ratio: float = 0.65,
        final_lr_frac: float = 0.05,
        init_lr_frac: float = 0.01,
        max_steps: int = 10000,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self._train_dataloader: DataLoader | None = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _default_loss(self, batch) -> torch.Tensor:
        """Next-token CE for a GPT whose forward(idx, targets) -> loss."""
        idx, targets = batch
        return self.model(idx, targets)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = (
            self.loss_fn(self.model, batch)
            if self.loss_fn is not None
            else self._default_loss(batch)
        )
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        """MinExpLSTM by default, AdamW baseline on request, plus the
        warmup + warmdown ``LambdaLR`` schedule.

        Subscript, not attribute, access on ``hparams``: Lightning's stub
        types the property as ``MutableMapping[str, Any]``, which has no
        attribute form, so every ``self.hparams.lr`` read was an
        unresolved-attribute error. Subscript returns ``Any`` and is
        what the dict-backed hyperparameter store actually supports.
        """
        param_groups = [dict(params=self.model.parameters(), lr=self.hparams["lr"])]
        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=self.hparams["weight_decay"]
            )
        else:
            optimizer = SGD_MinExpLstm_Optimizer(
                param_groups,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
                lam=self.hparams["minexp_lam"],
                clamp_val=self.hparams["minexp_clamp"],
                meta_lr=self.hparams["minexp_meta_lr"],
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            self._lr_lambda,
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    def _lr_lambda(self, step: int) -> float:
        """Warmup + warmdown multiplier, matching ``get_lr_multiplier``."""
        warmup = self.hparams["warmup_steps"]
        total = self.hparams["max_steps"]
        warmdown = round(self.hparams["warmdown_ratio"] * total)
        init_frac = self.hparams["init_lr_frac"]
        final_frac = self.hparams["final_lr_frac"]
        if step < warmup:
            frac = step / max(warmup, 1)
            return frac * 1.0 + (1 - frac) * init_frac
        if step <= total - warmdown:
            return 1.0
        progress = (total - step) / max(warmdown, 1)
        return progress * 1.0 + (1 - progress) * final_frac

    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None:
            raise RuntimeError(
                "attach a DataLoader with `module._train_dataloader = loader`"
            )
        return self._train_dataloader

    def tune_lr(self, trainer: Trainer, **lr_find_kwargs) -> float | None:
        """Run Lightning's automatic LR finder and apply the suggestion.

        Lightning 2.6 removed ``Trainer.tune()``; the tuner is a
        standalone ``Tuner(trainer)`` whose ``lr_find`` sweeps the LR
        and returns a ``_LRFinder`` whose ``suggestion()`` is the
        optimal initial LR. This applies that LR to the module and the
        optimizer's first param group so ``fit`` starts from it, and
        returns it for logging. Pass ``num_training`` / ``min_lr`` /
        ``max_lr`` via ``lr_find_kwargs``. Returns None if the finder
        produced no suggestion (e.g. the loss never decreased).
        """
        from lightning.pytorch.tuner import Tuner

        result = Tuner(trainer).lr_find(self, **lr_find_kwargs)
        assert result is not None, (
            "Tuner.lr_find returned None; pass a valid model and train_loader"
        )
        suggested = result.suggestion()
        self.hparams["lr"] = suggested
        # Apply the suggestion to the right LR. The SGD and MeZO-MinExpLSTM optimizers own an AdamW meta-optimizer (the "optimizer's optimizer" that trains the cell and the data-dependent LR predictor); its LR is a plain hyperparameter, so Lightning's finder optimizes IT before training starts (warmup). The primary optimizer's LR is data-dependent (the learnable predictor), so the finder does not set it directly -- it only seeds the predictor's initial bias. The AdamW baseline has no meta-optimizer, so the finder tunes its own (only) LR.
        primary = self.trainer.optimizers[0]
        meta_opt = getattr(primary, "meta_optimizer", None)
        if meta_opt is not None:
            for group in meta_opt.param_groups:
                group["lr"] = suggested
            self.hparams["meta_lr"] = suggested
        for group in primary.param_groups:
            group["lr"] = suggested
        return suggested


def make_lightning_trainer(
    max_steps: int = 10000,
    accelerator: str | None = None,
    devices: int | list | str = "auto",
    strategy: Strategy | str | None = None,
    logger: Logger | list[Logger] | None = None,
    callbacks: list | None = None,
    val_check_interval: float | int = 1.0,
    log_every_n_steps: int = 100,
    enable_progress_bar: bool = True,
    enable_checkpointing: bool = True,
    **trainer_kwargs,
) -> Trainer:
    """Build a ``Trainer`` for the MinExpLSTM wrapper.

    Attaches ``LearningRateMonitor`` so the warmup/warmdown schedule
    is visible in the logger, and enables Lightning's ``tune()``
    path (automatic LR finding) by keeping the trainer
    ``auto_lr_find``-ready. Pass ``auto_lr_find="lr"`` (or a
    callable) via ``trainer_kwargs`` to let ``Trainer.tune()``
    pick the LR before ``fit``.
    """
    cbs = list(callbacks or [])
    # LearningRateMonitor needs a logger to write to; without one it raises on_train_start. `logger=False` still yields a DummyLogger that LRMonitor rejects, so only attach the monitor when a real logger was passed.
    if logger not in (None, False):
        cbs.append(LearningRateMonitor(logging_interval="step"))
    # Only forward the flags the caller actually set: Lightning rejects `strategy=None` / `accelerator=None` as literal values, so a None means "let Lightning autodetect".
    kwargs = dict(
        max_steps=max_steps,
        devices=devices,
        logger=logger,
        callbacks=cbs,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=enable_checkpointing,
    )
    if accelerator is not None:
        kwargs["accelerator"] = accelerator
    if strategy is not None:
        kwargs["strategy"] = str(strategy)
    kwargs.update(trainer_kwargs)
    return Trainer(**kwargs)


class JointTrainableModule(LightningModule):
    """PRD section 5.3: target model + SGD-MinExpLSTM, manual optimization.

    The primary loss steps the target model through the FiLM-modulated
    SGD update; every ``meta_interval`` steps the loss is rebuilt
    differentiably from the updated parameters (via
    ``torch.func.functional_call``) and the cell co-trains through
    ``meta_step(model, closure)``.

    Why the closure form and not the PRD's ``meta_step(meta_loss)``
    shorthand: the primary ``step()`` runs under ``no_grad`` with
    in-place ``.data`` updates and a non-differentiable round to
    INT8, so a loss computed after the step carries no graph to the
    cell. The closure replays the update differentiably, which is
    what makes the meta-gradient real.
    """

    def __init__(
        self,
        target_model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        lam: float = 0.10,
        clamp_val: float = 1.0,
        meta_lr: float = 1e-3,
        meta_interval: int = 5,
    ) -> None:
        super().__init__()
        self.target_model = target_model
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["target_model"])
        self.meta_interval = meta_interval
        # Manual optimization: the meta-pass interleaves two backward/step pairs per training_step.
        self.automatic_optimization = False

    def forward(self, x):
        return self.target_model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        opts = self.optimizers()
        # `optimizers()` may hand back the singleton or a one-list; this module configures exactly one.
        opt = opts[0] if isinstance(opts, list) else opts
        assert isinstance(opt, SGD_MinExpLstm_Optimizer)
        x, y = batch

        # 1. Primary model forward, backward, and FiLM-modulated step.
        logits = self(x)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        self.manual_backward(loss)
        opt.step()

        # 2. Online meta-optimization pass every N steps. This MUST run before ``zero_grad``: the closure rebuilds each updated parameter from ``p.grad``, so a zeroed gradient leaves the closure with nothing to rebuild (the optimizer raises rather than silently skipping). The closure's own ``backward`` populates the model gradients, so clear them afterwards -- the next step's ``manual_backward`` would otherwise accumulate onto stale gradients.
        if (batch_idx + 1) % self.meta_interval == 0:
            loss_fn = self.loss_fn

            def _closure(params):
                out = torch.func.functional_call(self.target_model, params, (x,))
                return loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))

            opt.meta_step(self.target_model, _closure)

        opt.zero_grad()

        self.log("train_loss", loss.detach(), prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        opt = SGD_MinExpLstm_Optimizer(
            self.target_model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            lam=self.hparams["lam"],
            clamp_val=self.hparams["clamp_val"],
            meta_lr=self.hparams["meta_lr"],
        )
        return [opt]


class MeZoJointTrainableModule(LightningModule):
    """PRD v2.0 section 5: target model + MeZO-MinExpLSTM, manual optimization.

    The zeroth-order twin of ``JointTrainableModule``. Each
    ``training_step`` runs MeZO's dual forward passes -- the
    scalar SPSA estimate ``(loss_pos - loss_neg) / (2 *
    epsilon)`` is staged between them -- then applies the
    FiLM-modulated MeZO update through ``MeZo_MinExpLstm_Optimizer``
    (no gradient is materialized for the primary model). Every
    ``meta_interval`` steps the loss is rebuilt differentiably
    from the updated parameters (via ``torch.func.functional_call``)
    and the shared cell co-trains through ``meta_step``.

    Memory: the perturbation vectors are regenerated from the PRNG
    seeds, so the optimizer state is one INT8 hidden state per
    parameter plus the K-scalar seed/mu history -- the MeZO
    inference-equivalent footprint the PRD targets.
    """

    def __init__(
        self,
        target_model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        clamp_val: float = 1.0,
        meta_lr: float = 1e-3,
        meta_interval: int = 5,
        # MeZO-specific hyperparameters (PRD 2.1/2.2/2.5).
        epsilon: float = 1e-3,
        K: int = 4,
        # Intra-step multi-probe batch size (PRD v2.1).
        M: int = 1,
        delta_q: float = 0.0625,
        a_base: float = 1.5,
        a_min: float = 0.1,
        lr_max: float | None = None,
        backend: str = "cpu",
    ) -> None:
        super().__init__()
        self.target_model = target_model
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["target_model"])
        self.meta_interval = meta_interval
        # Manual optimization: the MeZO pass interleaves two forward evaluations and one meta-backward per step.
        self.automatic_optimization = False
        # The step seed counter: each training step draws a fresh seed, which both the perturbation replay and the FIR history consume (PRD 2.2).
        self._step_seed = 0

    def forward(self, x):
        return self.target_model(x)

    def _next_seed(self) -> int:
        """Draw the next per-step perturbation seed."""
        self._step_seed += 1
        return self._step_seed

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, list) else opts
        # ``self.optimizers()`` returns Lightning's ``LightningOptimizer`` wrapper, which delegates attribute access to the inner optimizer but is a distinct object. The MeZO optimizer stages its scalar SPSA estimate between the two forward passes as instance state, so the perturbation, staging, and step must all run on the *inner* optimizer -- otherwise the wrapper holds the staged estimate while the inner optimizer's ``step`` sees None.
        inner = getattr(opt, "_optimizer", opt)
        assert isinstance(inner, MeZo_MinExpLstm_Optimizer)
        x, y = batch
        M = inner.M

        # 1. MeZO dual forward passes. Single-probe (M=1, the v2.0 default): one perturb(+eps) -> evaluate -> perturb(-2eps) -> evaluate -> perturb(+eps) cycle. Multi-probe (M>1, PRD v2.1): M independent dual-forward cycles, one per probe seed, so the batch samples M directions around the same base state. Each cycle restores the baseline exactly, so the probes are independent and the estimator variance drops by 1/M (PRD v2.1 section 4.1).
        probe_seeds: list[int] = []
        losses_pos: list[torch.Tensor] = []
        losses_neg: list[torch.Tensor] = []
        for m in range(M):
            seed = self._next_seed()
            probe_seeds.append(seed)
            inner.perturb(seed, scaling=1.0)
            logits_pos = self(x)
            losses_pos.append(
                self.loss_fn(
                    logits_pos.reshape(-1, logits_pos.size(-1)),
                    y.reshape(-1),
                )
            )
            inner.perturb(seed, scaling=-2.0)
            logits_neg = self(x)
            losses_neg.append(
                self.loss_fn(
                    logits_neg.reshape(-1, logits_neg.size(-1)),
                    y.reshape(-1),
                )
            )
            inner.perturb(seed, scaling=1.0)

        # 2. Stage the SPSA estimate(s) and apply the FiLM-modulated MeZO update. No ``manual_backward``: MeZO is zeroth-order -- the perturbation already mutated the parameters in place, so the forward graph is stale and backward would raise. The scalar ``(loss_pos - loss_neg) / (2 * epsilon)`` (single-probe) or its batch mean (multi-probe) is the gradient surrogate; ``step`` consumes it. ``opt.step()`` (the Lightning wrapper) rather than ``inner.step()``: Lightning's manual- optimization loop signals step completion through the wrapper, and the wrapper delegates to the inner optimizer's ``step`` -- which sees the staged estimate because staging ran on ``inner``.
        if M == 1:
            inner.stage_mezo_estimate(probe_seeds[0], losses_pos[0], losses_neg[0])
        else:
            # The multi-probe staging averages the per-probe estimates (PRD v2.1 section 3.1, Option B): stack the M scalar losses into [M] and stage the batch mean.
            inner.stage_mezo_estimate_multi(
                probe_seeds,
                torch.stack(losses_pos),
                torch.stack(losses_neg),
            )
        opt.step()

        # 3. Online meta-optimization pass every N steps: rebuild the loss differentiably from the updated parameters so the cell's (gamma, beta) sit on the gradient path. The closure's backward populates the model gradients (the rebuilt update is on the graph); clear them so they do not accumulate across steps -- MeZO never consumes them (the primary update is zeroth-order, no manual_backward).
        if (batch_idx + 1) % self.meta_interval == 0:
            loss_fn = self.loss_fn

            def _closure(params):
                out = torch.func.functional_call(self.target_model, params, (x,))
                return loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))

            inner.meta_step(self.target_model, _closure)
            inner.zero_grad()

        # Log the batch-mean losses (the multi-probe estimate averages the per-probe scalars, so the logged loss is comparable across probe counts).
        mean_pos = torch.stack(losses_pos).mean()
        mean_neg = torch.stack(losses_neg).mean()
        self.log("train_loss", mean_pos.detach(), prog_bar=True)
        self.log("mezo_loss_pos", mean_pos.detach(), prog_bar=False)
        self.log("mezo_loss_neg", mean_neg.detach(), prog_bar=False)
        return mean_pos

    def configure_optimizers(self):  # type: ignore[override]
        opt = MeZo_MinExpLstm_Optimizer(
            self.target_model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            clamp_val=self.hparams["clamp_val"],
            meta_lr=self.hparams["meta_lr"],
            epsilon=self.hparams["epsilon"],
            K=self.hparams["K"],
            M=self.hparams["M"],
            delta_q=self.hparams["delta_q"],
            a_base=self.hparams["a_base"],
            a_min=self.hparams["a_min"],
            lr_max=self.hparams["lr_max"],
            backend=self.hparams["backend"],  # type: ignore[arg-type]
        )
        return [opt]
