"""Tests for Knowledge Distillation anchoring (PRD section 3.1)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.lcqat.kd import KDLoss, kd_loss


class TinyModel(nn.Module):
    def __init__(self, vocab=32, dim=8):
        super().__init__()
        self.w = nn.Linear(dim, vocab)

    def forward(self, x):
        return self.w(x)


def test_kd_loss_matches_kl_divergence():
    torch.manual_seed(0)
    teacher = torch.randn(4, 8)
    student = torch.randn(4, 8, requires_grad=True)
    loss = kd_loss(teacher, student, tau=2.0)
    # Expected: tau^2 * KL( softmax(T/tau) || softmax(S/tau) ), log_target form.
    # PRD 3.1: L_KD = tau^2 * KL( softmax(Z_teacher/tau) || softmax(Z_student/tau) )
    # F.kl_div(input, target, log_target=True) computes KL(target || input),
    # so KL(teacher || student) = F.kl_div(student_logp, teacher_logp).
    exp = 4.0 * F.kl_div(
        F.log_softmax(student.detach() / 2.0, dim=-1),
        F.log_softmax(teacher / 2.0, dim=-1),
        reduction="batchmean",
        log_target=True,
    )
    assert torch.allclose(loss, exp)
    assert loss.item() >= 0.0


def test_kd_loss_grad_flows_to_student_only():
    teacher = TinyModel()
    student = TinyModel()
    for p in teacher.parameters():
        p.requires_grad_(False)
    kd = KDLoss(teacher, alpha=0.5, tau=1.0)
    x = torch.randn(2, 8)
    loss = kd(student(x), x)
    loss.backward()
    # teacher params have no grad; student params get grad
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in student.parameters())


def test_kd_loss_alpha_zero_is_kl_only():
    teacher = TinyModel()
    student = TinyModel()
    for p in teacher.parameters():
        p.requires_grad_(False)
    kd = KDLoss(teacher, alpha=0.0, tau=1.0)
    x = torch.randn(2, 8)
    loss = kd(student(x), x)
    assert torch.isfinite(loss)


def test_kd_alpha_range_guard():
    teacher = TinyModel()
    with pytest.raises(ValueError):
        KDLoss(teacher, alpha=1.5)
    with pytest.raises(ValueError):
        KDLoss(teacher, alpha=-0.1)


def test_kd_teacher_frozen_on_init():
    teacher = TinyModel()
    before = [p.requires_grad for p in teacher.parameters()]
    KDLoss(teacher, alpha=0.3, tau=1.0)
    after = [p.requires_grad for p in teacher.parameters()]
    assert all(not a for a in after)
    assert any(a for a in before)


def test_kd_teacher_eval_mode():
    teacher = TinyModel()
    teacher.train()
    KDLoss(teacher, alpha=0.3, tau=1.0)
    assert not teacher.training
