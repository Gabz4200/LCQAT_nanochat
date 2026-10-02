"""
Knowledge Distillation (KD) anchoring for LC-QAT (PRD section 3.1).

High compression ratios (K=3 weights, K=15 activations) compress the loss
manifold into sharp local minima. LC-QAT anchors the student QAT optimization
using KL-divergence against the original unquantized FP32/BF16 teacher model:

    L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )
    L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD

The teacher is a frozen, detached copy of the model before quantization was
applied (or any other unquantized reference model). It is never optimized.

Two forms of the same anchor live here, because the two training objectives
have two different output spaces:

* `KDLoss` -- *logit* KD, for the next-token cross-entropy objective. The
  teacher is a full `GPT`, and the divergence is a KL over the vocabulary
  distribution. Incompatible with the EDM objective, which has no logits.
* `DenoiserDistiller` -- *denoiser* KD, for the DiffusionBlocks EDM objective.
  The teacher is a frozen float twin of the block engine, and the divergence is
  an EDM regression anchor against the twin's denoised embedding

      L_KD = w(sigma) * || D_q(x_noisy, sigma) - D_fp(x_noisy, sigma) ||^2
      L_b  = (1 - alpha) * w(sigma) * || D_q - clean ||^2 + alpha * L_KD

  Both terms see the *same* `noisy` input and the same sigma, so the anchor
  isolates the effect of quantization rather than of resampled noise. The twin
  holds only the active block's forward, so it costs L/B of a forward pass and
  no backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _frozen_teacher_forward(teacher, forward_fn: "callable"):
    """Run `forward_fn` against a frozen teacher and return an FP32 detached result.

    Single arbiter for "how the teacher is run": under `no_grad`, in eval mode,
    with the caller's training mode restored afterwards, and never part of the
    optimizer graph -- so no gradient can leak back into the teacher and no
    dropout/BN statistic can be perturbed by the student's own training mode.

    `teacher` is duck-typed rather than `nn.Module`-typed because the EDM
    teacher is a `DiffusionBlockEngine`, which is deliberately not an
    `nn.Module` (it owns three of them).
    """
    was_training = getattr(teacher, "training", False)
    if hasattr(teacher, "eval"):
        teacher.eval()
    try:
        out = forward_fn()
    finally:
        if hasattr(teacher, "train"):
            teacher.train(was_training)
    return out.detach().to(torch.float32)


@torch.no_grad()
def _teacher_logits(model: torch.nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Run the frozen teacher forward and return FP32 logits."""
    return _frozen_teacher_forward(model, lambda: model(idx))


def denoiser_kd_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM-weighted squared error between the quantized and float denoisers.

    The denoiser's output is a clean *embedding*, not a distribution, so the
    natural divergence here is the regression error the EDM objective already
    minimizes, re-targeted from `clean` onto the float teacher's prediction:

        L_KD = w(sigma) * || D_q(x_noisy, sigma) - D_fp(x_noisy, sigma) ||^2

    The same `w(sigma)` weighting as the EDM term is applied, so the anchor has
    the same noise-range emphasis as the objective it anchors: a low-sigma step
    is not pushed toward the teacher harder than the data itself demands.

    Args:
        student_pred: quantized denoiser output (B, T, n_embd), in the graph.
        teacher_pred: float teacher output, same shape, already detached.
        weight: EDM `w(sigma)` scalar. `None` means an unweighted mean.

    Returns:
        scalar tensor, differentiable w.r.t. `student_pred` only.
    """
    se = (student_pred - teacher_pred).square()
    return se.mean() if weight is None else (weight * se).mean()


def kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    tau: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL-divergence between teacher and student softmax distributions.

    Args:
        teacher_logits: FP32 logits from the frozen teacher, shape (..., V).
        student_logits: logits from the quantized student, same shape.
        tau: softmax temperature; higher tau softens both distributions.
        reduction: "batchmean" | "mean" | "sum" | "none".

    Returns:
        tau^2 * D_KL( softmax(Z_teacher/tau) || softmax(Z_student/tau) ),
        reduced per `reduction`. The tau^2 prefactor matches Polino et al.
        so the gradient magnitude stays tau-invariant.
    """
    teacher_logp = F.log_softmax(teacher_logits / tau, dim=-1)
    student_logp = F.log_softmax(student_logits / tau, dim=-1)
    kl = F.kl_div(student_logp, teacher_logp, reduction=reduction, log_target=True)
    return tau * tau * kl


class KDLoss:
    """Stateful KD wrapper used by the training loop.

    Holds the frozen teacher model and the anchor hyperparameters. The
    teacher is expected to be a *separate* model instance (e.g. a float
    snapshot of the pre-QAT model) so that `kd_loss` is a pure function of
    the two logit tensors.
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        alpha: float = 0.1,
        tau: float = 1.0,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.teacher = teacher
        self.alpha = float(alpha)
        self.tau = float(tau)
        # Freeze the teacher: no gradients, no optimizer state.
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def __call__(
        self,
        student_logits: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the KD term for one batch.

        Args:
            student_logits: quantized student logits (B, T, V).
            idx: input token ids, used to run the frozen teacher.

        Returns:
            scalar KD loss tensor (reduced over the batch).
        """
        teacher_logits = _teacher_logits(self.teacher, idx)
        return kd_loss(teacher_logits, student_logits, tau=self.tau)


class DenoiserDistiller:
    """Denoiser distillation: float twin -> quantized denoiser (EDM objective).

    `KDLoss` anchors a next-token cross-entropy against the teacher's
    vocabulary distribution. The EDM objective has no logits -- it regresses a
    denoised *embedding* -- so the same anchor has to be expressed in the
    denoiser's own output space. This class supplies that form: a frozen float
    twin of the block engine, run on the *same* noised input and the same sigma
    as the student, and its prediction used as the regression target in place of
    (weighted alongside) the clean embeddings.

        L_b = (1 - alpha) * w(sigma) * ||D_q - clean||^2
            + alpha     * w(sigma) * ||D_q - D_fp||^2

    The twin is an independent object, never a view into the student's
    parameters, so `alpha` cannot silently become a self-distillation no-op:
    with `alpha = 1` the loss is exactly zero only when the student's weights
    are bit-identical to the twin's.

    The teacher is frozen the same way `KDLoss` freezes its own: no grads, no
    optimizer state, eval mode. `DiffusionBlockEngine.parameters()` is used for
    the freeze sweep because the engine is not an `nn.Module` and its parameters
    live in three separate subtrees.
    """

    def __init__(
        self,
        teacher,
        alpha: float = 0.1,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.teacher = teacher
        self.alpha = float(alpha)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def __call__(
        self,
        student_pred: torch.Tensor,
        teacher_pred_fn: "callable",
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the denoiser KD term for one micro-step.

        Args:
            student_pred: quantized denoiser output (B, T, n_embd), in the graph.
            teacher_pred_fn: zero-arg callable returning the float twin's
                prediction for the *same* input. A callable rather than a
                tensor because the twin's forward only runs the active block,
                and calling it eagerly at the wrong point would either cost a
                forward nobody uses or evaluate a stale input.
            weight: EDM `w(sigma)` scalar, applied to both terms.

        Returns:
            scalar KD loss tensor, differentiable w.r.t. `student_pred`.
        """
        teacher_pred = _frozen_teacher_forward(self.teacher, teacher_pred_fn)
        return denoiser_kd_loss(student_pred, teacher_pred, weight=weight)
