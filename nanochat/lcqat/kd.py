"""
Knowledge Distillation (KD) anchoring for LC-QAT (PRD section 3.1).

High compression ratios (K=3 weights, K=15 activations) compress the loss
manifold into sharp local minima. LC-QAT anchors the student QAT optimization
using KL-divergence against the original unquantized FP32/BF16 teacher model:

    L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )
    L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD

The teacher is a frozen, detached copy of the model before quantization was
applied (or any other unquantized reference model). It is never optimized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _teacher_logits(model: torch.nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Run the frozen teacher forward and return FP32 logits.

    The teacher is run under torch.no_grad() and is never part of the
    optimizer graph, so gradients never leak back into it.
    """
    was_training = model.training
    model.eval()
    try:
        logits = model(idx)
    finally:
        model.train(was_training)
    return logits.detach().to(torch.float32)


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
