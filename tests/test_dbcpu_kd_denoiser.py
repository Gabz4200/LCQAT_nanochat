"""Denoiser distillation: the EDM-native form of the KD anchor (PRD 3.1).

`KDLoss` anchors a next-token cross-entropy against the teacher's vocabulary
distribution. The EDM objective has no logits -- it regresses a denoised
*embedding* -- which is why `--kd-alpha` + `--db-objective edm` refuses to start
rather than silently skipping. `DenoiserDistiller` is the anchor for that
objective: a frozen float twin of the block engine supplies the target, and
`denoise_step` mixes it into the objective.

What these tests hold the implementation to, in the order that matters:

1. The term is real. Zero when teacher and student agree exactly, strictly
   positive when they differ, and it tracks the size of the gap.
2. It is actually in the objective. The loss `denoise_step` returns differs with
   the distiller attached vs detached, by the documented convex mix.
3. It is differentiable into the *student's* parameters -- including the LC-QAT
   codebooks, which are the whole point of anchoring a quantized denoiser, and
   would be missed by a test that only checks matrix weights.

Point 2 is the one that would catch the failure mode that matters: a KD term
computed, logged, and then dropped on the floor before the backward.
"""

import copy

import pytest
import torch

from nanochat.models.quant.kd import DenoiserDistiller, denoiser_kd_loss
from nanochat.models.quant.retrofit import DEFAULT_PRESET, PRESETS, retrofit_model
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from tests.conftest import build_active_tiny_gpt


def _make_engine(num_blocks: int = 2) -> DiffusionBlockEngine:
    """Engine over a tiny GPT whose zero-init layers are randomized.

    The denoise heads and adapter output layers are zero at init, so an
    untrained engine's prediction is identically zero and *every* comparison
    here -- zero-vs-nonzero, with-vs-without the anchor -- would pass
    vacuously.
    """
    torch.manual_seed(0)
    engine = DiffusionBlockEngine(
        build_active_tiny_gpt(),
        EquiProbabilityPartitioner(num_blocks=num_blocks),
        dtype=torch.float32,
    )
    with torch.no_grad():
        for head in engine.denoise_heads:
            head.weight.normal_(std=0.05)
            head.bias.normal_(std=0.05)
        for adapter in engine.adapters:
            adapter.mlp[-1].weight.normal_(std=0.05)
    return engine


def _retrofitted_engine() -> DiffusionBlockEngine:
    """LC-QAT engine, so the KD gradient can be asserted on codebook deltas."""
    torch.manual_seed(0)
    engine = DiffusionBlockEngine(
        retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET]),
        EquiProbabilityPartitioner(num_blocks=2),
        dtype=torch.float32,
    )
    with torch.no_grad():
        for head in engine.denoise_heads:
            head.weight.normal_(std=0.05)
            head.bias.normal_(std=0.05)
        for adapter in engine.adapters:
            adapter.mlp[-1].weight.normal_(std=0.05)
    return engine


def _float_twin(engine: DiffusionBlockEngine) -> DiffusionBlockEngine:
    """Deep copy with every weight perturbed, standing in for a float twin.

    A twin that is bit-identical to the student would make every assertion here
    pass for the wrong reason, so the perturbation is deliberate: it is the
    quantization gap the anchor is meant to measure.
    """
    twin = copy.deepcopy(engine)
    with torch.no_grad():
        for p in twin.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return twin


def test_when_student_matches_teacher_then_the_kd_term_is_zero() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 5, 8)
    loss = denoiser_kd_loss(pred, pred.clone())
    assert loss.item() == 0.0


def test_when_student_differs_then_the_kd_term_is_positive_and_tracks_the_gap() -> None:
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 8)
    small = denoiser_kd_loss(teacher + 0.01, teacher).item()
    large = denoiser_kd_loss(teacher + 0.10, teacher).item()
    assert small > 0.0
    # Squared error, so a 10x perturbation must give ~100x the loss.
    assert large == pytest.approx(small * 100.0, rel=1e-4)


def test_when_weighted_then_the_edm_scaling_is_applied() -> None:
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 8)
    unweighted = denoiser_kd_loss(teacher + 0.1, teacher).item()
    weighted = denoiser_kd_loss(
        teacher + 0.1, teacher, weight=torch.tensor(16.0)
    ).item()
    assert weighted == pytest.approx(16.0 * unweighted, rel=1e-6)


def test_when_attached_then_the_returned_loss_changes_by_the_documented_mix() -> None:
    """The acceptance property: the KD term reaches the objective.

    With `distiller=None` the returned loss is exactly the EDM data term; with
    alpha attached it is `(1-alpha)*edm + alpha*kd`. Dropping the term before
    the backward would leave both equal, so this fails.
    """
    idx = torch.randint(0, 128, (2, 16))
    engine = _make_engine()
    twin = _float_twin(engine)
    alpha = 0.3

    torch.manual_seed(7)
    plain, _ = engine.denoise_step(idx, block_idx=0)

    engine.set_distiller(DenoiserDistiller(twin, alpha=alpha))
    torch.manual_seed(7)
    anchored, _ = engine.denoise_step(idx, block_idx=0)

    kd = engine.last_kd_loss
    assert kd > 0.0, "the twin differs from the student, so the anchor must be > 0"
    expected = (1.0 - alpha) * plain.item() + alpha * kd
    assert anchored.item() == pytest.approx(expected, rel=1e-5)
    assert anchored.item() != pytest.approx(plain.item(), rel=1e-9)


def test_when_distiller_absent_then_no_kd_term_is_computed() -> None:
    idx = torch.randint(0, 128, (2, 16))
    engine = _make_engine()
    assert engine.distiller is None
    torch.manual_seed(7)
    loss, _ = engine.denoise_step(idx, block_idx=0)
    assert engine.last_kd_loss == 0.0
    assert torch.isfinite(loss)


def test_when_alpha_one_then_the_loss_is_purely_the_teacher_anchor() -> None:
    """alpha=1 drops the data term entirely: the objective becomes the anchor."""
    idx = torch.randint(0, 128, (2, 16))
    engine = _make_engine()
    twin = _float_twin(engine)
    engine.set_distiller(DenoiserDistiller(twin, alpha=1.0))
    torch.manual_seed(7)
    loss, _ = engine.denoise_step(idx, block_idx=0)
    assert loss.item() == pytest.approx(engine.last_kd_loss, rel=1e-6)


def test_when_identical_twin_then_the_kd_term_vanishes() -> None:
    """A twin carrying the student's weights would make the flag a no-op.

    This is the guard against `--kd-denoiser-alpha` silently degenerating into
    self-distillation: with an exactly-copied twin the anchor must read zero,
    which is exactly why the real twin must be built float (see
    `build_float_twin`).
    """
    idx = torch.randint(0, 128, (2, 16))
    engine = _make_engine()
    engine.set_distiller(DenoiserDistiller(copy.deepcopy(engine), alpha=1.0))
    loss, _ = engine.denoise_step(idx, block_idx=0)
    assert engine.last_kd_loss == pytest.approx(0.0, abs=1e-9)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_when_denoising_with_kd_then_gradients_reach_the_codebooks() -> None:
    """Differentiability, checked where it matters.

    Matrix weights are the easy case -- a dense grad there proves little about an
    anchor whose purpose is to train quantization parameters. The assertion that
    matters is on `raw_pos_deltas` / `raw_neg_deltas`, the LC-QAT codebook
    parameters: if the KD term were computed outside the student's graph, those
    would carry a matrix-weight gradient at best and none at all.
    """
    idx = torch.randint(0, 128, (2, 16))
    engine = _retrofitted_engine()
    twin = _float_twin(engine)
    engine.set_distiller(DenoiserDistiller(twin, alpha=1.0))

    torch.manual_seed(11)
    loss, _ = engine.denoise_step(idx, block_idx=0)
    loss.backward()

    codebook_grads = {
        n: p.grad
        for n, p in engine.named_parameters()
        if ("raw_pos_deltas" in n or "raw_neg_deltas" in n) and p.grad is not None
    }
    assert codebook_grads, "the LC-QAT engine should own codebook deltas"
    # Magnitude, not presence: at alpha=1 the data term is scaled by zero, so a
    # `.grad` of exactly 0.0 on a codebook means the anchor was detached from the
    # graph and the whole objective reduced to a constant. `is not None` would
    # pass on that, which is the whole reason this asserts `> 0`.
    assert any(g.abs().sum() > 0 for g in codebook_grads.values()), (
        "no codebook received a nonzero gradient from the KD anchor"
    )

    # And the teacher received nothing: it is frozen, not merely detached.
    assert all(p.grad is None for p in twin.parameters())


def test_when_backpropagating_then_no_teacher_graph_is_retained() -> None:
    """The twin must contribute a constant, not a second optimization target.

    `alpha=1` means "match the teacher exactly", so if any gradient path ran
    back into the twin, the twin -- not the student -- would absorb the update.
    The twin's parameters having no `.grad` after the backward is the check.
    """
    idx = torch.randint(0, 128, (2, 16))
    engine = _make_engine()
    twin = _float_twin(engine)
    engine.set_distiller(DenoiserDistiller(twin, alpha=1.0))
    loss, _ = engine.denoise_step(idx, block_idx=0)
    loss.backward()
    assert all(p.grad is None for p in twin.parameters())
    assert all(not p.requires_grad for p in twin.parameters())
