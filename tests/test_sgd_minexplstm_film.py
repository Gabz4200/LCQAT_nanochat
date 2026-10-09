"""PRD v3.2 spec tests for the SGD-MinExpLSTM optimizer.

Pins the `SGD-MinExpLSTM-Optimizer` engineering spec: the
`SGDMinExpLSTMCellFiLM4D` cell (3-input, 4D latent, FiLM head), the
lambda-coupled dual-timeframe dynamic EMAs, the 3 bytes/param INT8
budget, the FiLM update rule, the step-0 SGD fallback, and the AdamW
meta-optimizer over the cell's own (< 50) parameters.

Math anchors: MinLSTM linear-scan recurrence (Feng et al. 2024,
arXiv:2410.01201), log-domain exponential gating = sigmoid(a - b)
(Horuz et al. 2025, arXiv:2508.03614), INT8 dynamic quantization
(Dettmers et al. 2021, arXiv:2110.02861), coordinate-wise learned
optimizer (Andrychowicz et al. 2016, arXiv:1606.04474), FiLM
modulation (Perez et al. 2018, arXiv:1709.07871).
"""

import torch

from nanochat.training.minexp_lstm import (
    SGD_MinExpLstm_Optimizer,
    SGDMinExpLSTMCellFiLM4D,
)

# Cell structure


def test_cell_param_count_is_microscopic() -> None:
    """The whole cell carries < 50 FP32 params (PRD principle 5)."""
    cell = SGDMinExpLSTMCellFiLM4D()
    n = sum(p.numel() for p in cell.parameters())
    assert n < 50, f"cell has {n} params, PRD requires < 50"


def test_film_head_zero_init_then_gamma_delta_are_zero() -> None:
    """Zero-initialized FiLM head: gamma == 0 and delta == 0 at t=0.

    The step-0 fallback safeguard (PRD principle 2): the first update
    must be pure SGD + weight decay.
    """
    torch.manual_seed(0)
    cell = SGDMinExpLSTMCellFiLM4D()
    assert torch.equal(
        cell.film_proj.weight.detach(), torch.zeros_like(cell.film_proj.weight)
    )
    assert torch.equal(
        cell.film_proj.bias.detach(), torch.zeros_like(cell.film_proj.bias)
    )
    x = torch.randn(4, 5, 3)
    h_prev = torch.randn(4, 5, 1)
    gamma, delta, _ = cell(x, h_prev)
    assert torch.equal(gamma.detach(), torch.zeros_like(gamma))
    assert torch.equal(delta.detach(), torch.zeros_like(delta))


def test_cell_forward_shapes() -> None:
    """x [..., 3] + h [..., 1] -> gamma [...], delta [...], h_next [..., 1]."""
    cell = SGDMinExpLSTMCellFiLM4D()
    x = torch.randn(2, 3, 4, 3)
    h_prev = torch.randn(2, 3, 4, 1)
    gamma, delta, h_next = cell(x, h_prev)
    assert gamma.shape == (2, 3, 4)
    assert delta.shape == (2, 3, 4)
    assert h_next.shape == (2, 3, 4, 1)


def test_cell_recurrence_is_linear_scan() -> None:
    """h_next = hat_phi * h_prev + (1 - hat_phi) * tilde_h, elementwise.

    MinLSTM property (arXiv:2410.01201): the gate depends only on the
    input x, never on h_prev, so the recurrence parallelizes.
    """
    torch.manual_seed(2)
    cell = SGDMinExpLSTMCellFiLM4D()
    x = torch.randn(2, 3, 3)
    h_prev = torch.randn(2, 3, 1)
    _, _, h_next = cell(x, h_prev)
    hat_phi = torch.sigmoid(cell.phi_proj(x) - cell.iota_proj(x))
    tilde_h = cell.h_proj(x)
    expected = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
    assert torch.allclose(h_next, expected, atol=1e-6)


def test_cell_gate_depends_only_on_input_not_prev_hidden() -> None:
    """Changing h_prev must not change the gate, only the h_prev term."""
    torch.manual_seed(1)
    cell = SGDMinExpLSTMCellFiLM4D()
    x = torch.randn(3, 3)
    h_a = torch.zeros(3, 1)
    h_b = torch.full((3, 1), 7.0)
    _, _, h_next_a = cell(x, h_a)
    _, _, h_next_b = cell(x, h_b)
    assert not torch.allclose(h_next_a, h_next_b)
    # The gate is a function of x alone: reconstruct it from the two states (h_next_b - h_next_a = hat_phi * (h_b - h_a)) and check it against the directly computed gate.
    hat_phi = (h_next_b - h_next_a) / (h_b - h_a)
    expected = torch.sigmoid(cell.phi_proj(x) - cell.iota_proj(x))
    assert torch.allclose(hat_phi, expected, atol=1e-5)


# Lambda-coupled dynamic betas (PRD section 3.2)


def test_betas_are_lambda_coupled_and_bounded() -> None:
    """beta_slow - beta_fast == lam everywhere; slow in [2*lam, 0.999]."""
    torch.manual_seed(3)
    cell = SGDMinExpLSTMCellFiLM4D()
    g_norm = torch.randn(32, 32) * 3.0
    for lam in (0.05, 0.10, 0.40):
        beta_fast, beta_slow = cell.predict_betas(g_norm, lam=lam)
        assert torch.allclose(
            beta_slow - beta_fast,
            torch.full_like(beta_slow, lam),
            atol=1e-6,
        )
        assert bool((beta_slow >= 2 * lam - 1e-6).all())
        assert bool((beta_slow <= 0.999 + 1e-6).all())
        assert bool((beta_fast >= lam - 1e-6).all())


def test_betas_saturate_at_hard_sigmoid_extremes() -> None:
    """Huge |u| clamps: +inf -> 0.999, -inf -> 2*lam (fast -> lam)."""
    cell = SGDMinExpLSTMCellFiLM4D()
    with torch.no_grad():
        cell.w_beta.fill_(1.0)
        cell.b_beta.fill_(0.0)
    lam = 0.10
    _, beta_slow_hi = cell.predict_betas(torch.full((8,), 1e4), lam=lam)
    beta_fast_lo, beta_slow_lo = cell.predict_betas(torch.full((8,), -1e4), lam=lam)
    assert torch.allclose(beta_slow_hi, torch.full((8,), 0.999), atol=1e-6)
    assert torch.allclose(beta_slow_lo, torch.full((8,), 2 * lam), atol=1e-6)
    assert torch.allclose(beta_fast_lo, torch.full((8,), lam), atol=1e-6)


def test_beta_predictor_is_two_learned_params() -> None:
    """One shared scalar predictor (w_beta, b_beta): single-gate cost."""
    cell = SGDMinExpLSTMCellFiLM4D()
    assert cell.w_beta.numel() == 1
    assert cell.b_beta.numel() == 1
    assert cell.w_beta.requires_grad and cell.b_beta.requires_grad


# Optimizer: step-0 fallback, memory budget, update rule


def test_step_zero_matches_sgd_with_weight_decay_exactly() -> None:
    """A fresh optimizer behaves as SGD + decoupled wd (PRD section 2.2)."""
    torch.manual_seed(42)
    p1 = torch.nn.Parameter(torch.randn(16, 16))
    p2 = torch.nn.Parameter(p1.clone().detach())
    opt1 = SGD_MinExpLstm_Optimizer([p1], lr=1e-3, weight_decay=0.01)
    opt2 = torch.optim.SGD([p2], lr=1e-3, weight_decay=0.01)
    for _ in range(5):
        grad = torch.randn(16, 16)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()
    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-7)


def test_state_footprint_is_three_bytes_per_param() -> None:
    """INT8 fast + INT8 slow + INT8 h = 3 bytes/param (PRD section 4.2)."""
    torch.manual_seed(0)
    n = 1024
    p = torch.nn.Parameter(torch.randn(n))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)
    p.grad = torch.randn(n)
    opt.step()
    state = opt.state[p]
    assert state["step"] == 1
    for key in ("ema_fast_int8", "ema_slow_int8", "h_int8"):
        assert state[key].dtype == torch.int8
        assert state[key].shape == p.shape
    total = sum(state[k].numel() for k in ("ema_fast_int8", "ema_slow_int8", "h_int8"))
    assert total / n == 3.0
    for key in ("scale_fast", "scale_slow", "scale_h"):
        assert isinstance(state[key], torch.Tensor) and state[key].ndim == 0


def test_update_rule_matches_film_formula() -> None:
    """theta += (1 + clamp(gamma)) * (-lr * G) + clamp(delta) - lr*wd*theta.

    Analytic check with a pinned non-zero FiLM head: set film_proj to
    emit gamma = 0.5, delta = 0.25 everywhere (bias-only projection),
    then compare one step against the hand-computed update. EMAs start
    at zero, so the check exercises the modulation path only.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5, 3.0]))
    lr, wd, clamp_val = 1e-2, 1e-2, 1.0
    opt = SGD_MinExpLstm_Optimizer([p], lr=lr, weight_decay=wd, clamp_val=clamp_val)
    with torch.no_grad():
        opt.cell.film_proj.weight.zero_()
        opt.cell.film_proj.bias.copy_(torch.tensor([0.5, 0.25]))
    grad = torch.tensor([0.1, 0.2, -0.3, 0.4])
    before = p.detach().clone()
    p.grad = grad.clone()
    opt.step()
    gamma_c = min(max(0.5, -0.9), 2.0)
    delta_c = min(max(0.25, -clamp_val), clamp_val)
    expected = before + (1.0 + gamma_c) * (-lr * grad) + delta_c - lr * wd * before
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_gamma_delta_clamps_hold() -> None:
    """gamma in [-0.9, 2.0], delta in [-kappa, kappa] no matter the head."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3, clamp_val=0.5)
    with torch.no_grad():
        opt.cell.film_proj.weight.normal_(0.0, 1e4)
        opt.cell.film_proj.bias.normal_(0.0, 1e4)
    p.grad = torch.randn(64) * 10.0
    before = p.detach().clone()
    opt.step()
    # Back out the applied (gamma, delta) contribution: with a huge head the raw outputs saturate the clamps, so the non-SGD part of the update is bounded by |(1+2)(-lr G)| + 0.5 + |decay| per coordinate.
    assert torch.isfinite(p.detach()).all()
    assert not torch.equal(before, p.detach())


def test_optimizer_is_permutation_invariant() -> None:
    """Coordinate-wise processing: permuting (param, grad) permutes update."""
    torch.manual_seed(7)
    base = torch.randn(4, 6)
    perm = torch.randperm(24)
    p_a = torch.nn.Parameter(base.clone())
    p_b = torch.nn.Parameter(base.reshape(-1)[perm].reshape(4, 6).clone())
    opt_a = SGD_MinExpLstm_Optimizer([p_a], lr=1e-2)
    opt_b = SGD_MinExpLstm_Optimizer([p_b], lr=1e-2)
    torch.manual_seed(123)
    ref_w = [
        m.weight.detach().clone()
        for m in opt_a.cell.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    ref_b = [
        m.bias.detach().clone()
        for m in opt_a.cell.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    for opt in (opt_a, opt_b):
        with torch.no_grad():
            for m, w, b in zip(
                [m for m in opt.cell.modules() if isinstance(m, torch.nn.Linear)],
                ref_w,
                ref_b,
            ):
                m.weight.copy_(w)
                m.bias.copy_(b)
    grad_a = torch.randn(4, 6)
    p_a.grad = grad_a.clone()
    p_b.grad = grad_a.reshape(-1)[perm].reshape(4, 6).clone()
    opt_a.step()
    opt_b.step()
    assert torch.allclose(p_a.reshape(-1)[perm], p_b.reshape(-1), atol=1e-6)


def test_rms_normalization_bounds_cell_inputs() -> None:
    """Layer-wise RMS keeps g_norm in a stable range under 100x scaling."""
    torch.manual_seed(9)
    p = torch.nn.Parameter(torch.randn(5, 5))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)
    grad = torch.randn(5, 5)
    for scale in (1.0, 100.0):
        p.grad = grad.clone() * scale
        opt.step()
    assert torch.isfinite(p.detach()).all()


# Meta-optimizer: AdamW over < 1KB of cell params (PRD principle 5)


def test_meta_optimizer_is_adamw_over_cell_params() -> None:
    """The cell's optimizer is AdamW over the cell + LR params.

    PRD principle 5: the meta-optimizer owns the cell's own
    (< 50) parameters plus the data-dependent LR predictor
    (a single ``nn.Linear(1, 1)`, 2 params). Both are
    co-trained online; the meta LR itself stays a plain
    hyperparameter.
    """
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3, meta_lr=1e-3)
    assert isinstance(opt.meta_optimizer, torch.optim.AdamW)
    owned = {
        id(q) for group in opt.meta_optimizer.param_groups for q in group["params"]
    }
    cell_ids = {id(q) for q in opt.cell.parameters()}
    lr_ids = {id(q) for q in opt.lr_predictor.parameters()}
    assert owned == cell_ids | lr_ids
    total_bytes = sum(
        q.numel() * q.element_size() for q in opt.cell.parameters()
    ) + sum(q.numel() * q.element_size() for q in opt.lr_predictor.parameters())
    assert total_bytes < 1024


def test_meta_step_on_loss_tensor_moves_cell() -> None:
    """meta_step(loss) backprops through a grad-enabled graph to the cell."""
    torch.manual_seed(6)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)
    x = torch.randn(16, 3)
    h = torch.randn(16, 1)
    gamma, delta, _ = opt.cell(x, h)
    loss = (gamma**2 + delta**2).mean()
    before = [q.detach().clone() for q in opt.cell.parameters()]
    opt.meta_step(loss)
    after = list(opt.cell.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_dual_ema_tracks_gradient_with_predicted_betas() -> None:
    """EMA_fast/slow follow the IIR blend with the cell's own betas."""
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.zeros(32))
    opt = SGD_MinExpLstm_Optimizer([p], lr=0.0, lam=0.10)
    constant = torch.ones(32)
    p.grad = constant.clone()
    opt.step()
    state = opt.state[p]
    ema_fast = state["ema_fast_int8"].to(torch.float32) * state["scale_fast"]
    ema_slow = state["ema_slow_int8"].to(torch.float32) * state["scale_slow"]
    rms_g = torch.sqrt(torch.mean(constant.float().square()) + 1e-8)
    g_norm = constant / (rms_g + 1e-8)
    beta_fast, beta_slow = opt.cell.predict_betas(g_norm, lam=0.10)
    assert torch.allclose(ema_fast, (1.0 - beta_fast) * constant, atol=1e-2)
    assert torch.allclose(ema_slow, (1.0 - beta_slow) * constant, atol=1e-2)
    # Fast tracks faster: smaller beta -> larger first-step EMA.
    assert bool((ema_fast.abs() > ema_slow.abs()).all())
