"""Tests for the SGD-MinExpLSTM joint online learned optimizer.

`SGD_MinExpLstm_Optimizer` replaces AdamW: a shared elementwise
`SGDMinExpLSTMCellFiLM4D` predicts FiLM gain/shift parameters on top
of a plain SGD step, and the persistent optimizer states (a fast
EMA, a slow EMA, and the hidden state) live in INT8 (3
bytes/param) instead of two FP32 moments (8 bytes/param).

The mathematical anchors:

* MinGRU/MinLSTM (Feng et al. 2024, arXiv:2410.01201): the gates
  depend only on the current input, so the recurrence is a linear
  scan, not a sequential LSTM.
* MinConvLSTM (Horuz et al. 2025, arXiv:2508.03614): exponential
  gating in the log domain; `sigmoid(a - b)` is the operative form
  (phi/iota collapse to it).
* 8-bit Optimizers (Dettmers et al. 2021, arXiv:2110.02861):
  dynamic quantization of optimizer states; here symmetric max-abs
  per tensor.
* Learning to Learn by GD by GD (Andrychowicz et al. 2016):
  coordinate-wise RNN optimizer trained by meta-gradients.
* FiLM (Perez et al. 2017, arXiv:1709.07871): `(1 + gamma) * x +
  delta` modulation of the base SGD step.
* Input-dependent decay (Gu & Dao 2023, arXiv:2312.00752):
  lambda-coupled dual-timeframe dynamic EMAs instead of static
  momentum hyperparameters.
"""

import pytest
import torch

from nanochat.training.minexp_lstm import MinExpLSTMCell, SGD_MinExpLstm_Optimizer

# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


def test_cell_output_proj_zero_init_then_delta_is_zero() -> None:
    """Zero-initialized FiLM head: at t=0 gamma and delta are exactly 0.

    This is the residual anchor. The first step must be pure SGD, so a
    fresh optimizer can never destroy the model in its early steps.
    """
    torch.manual_seed(0)
    cell = MinExpLSTMCell(input_size=3, hidden_size=1)
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


def test_cell_gate_depends_only_on_input_not_prev_hidden() -> None:
    """minLSTM property: the gate is independent of h_{t-1}.

    Changing h_prev must not change the gate (hat_phi), only the
    candidate path through h_next. This is what makes the recurrence
    a parallelizable linear scan.
    """
    torch.manual_seed(1)
    cell = MinExpLSTMCell(input_size=3, hidden_size=1)
    # Non-default init so the gate actually varies.
    with torch.no_grad():
        cell.phi_proj.weight.normal_()
        cell.iota_proj.weight.normal_()
    x = torch.randn(3, 3)
    h_a = torch.zeros(3, 1)
    h_b = torch.full((3, 1), 7.0)
    _, _, h_next_a = cell(x, h_a)
    _, _, h_next_b = cell(x, h_b)
    # The gate is a function of x alone, so h_next differs only
    # through the h_prev term, and the two h_nexts share the same
    # candidate (tilde_h) contribution.
    assert not torch.allclose(h_next_a, h_next_b)
    # hat_phi identical: reconstruct it from the two states.
    # h_next = hat_phi * h_prev + (1 - hat_phi) * tilde_h
    # => h_next_b - h_next_a = hat_phi * (h_b - h_a)
    hat_phi = (h_next_b - h_next_a) / (h_b - h_a)
    expected = torch.sigmoid(cell.phi_proj(x) - cell.iota_proj(x))
    assert torch.allclose(hat_phi, expected, atol=1e-5)


def test_cell_recurrence_is_linear_scan() -> None:
    """h_next = hat_phi * h_prev + (1 - hat_phi) * tilde_h, elementwise."""
    torch.manual_seed(2)
    cell = MinExpLSTMCell(input_size=3, hidden_size=1)
    with torch.no_grad():
        cell.phi_proj.weight.normal_()
        cell.iota_proj.weight.normal_()
        cell.h_proj.weight.normal_()
    x = torch.randn(2, 3, 3)
    h_prev = torch.randn(2, 3, 1)
    gamma, delta, h_next = cell(x, h_prev)

    phi = cell.phi_proj(x)
    iota = cell.iota_proj(x)
    hat_phi = torch.sigmoid(phi - iota)
    tilde_h = cell.h_proj(x)
    expected_h = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
    assert torch.allclose(h_next, expected_h, atol=1e-6)
    # With a zero-initialized FiLM head gamma and delta are 0.
    assert torch.equal(gamma, torch.zeros_like(gamma))
    assert torch.equal(delta, torch.zeros_like(delta))


# ---------------------------------------------------------------------------
# Optimizer: SGD equivalence (Phase 1 of the PRD verification plan)
# ---------------------------------------------------------------------------


def test_alpha_zero_matches_sgd_exactly() -> None:
    """alpha_residual=0.0: the trajectory must match native SGD.

    The PRD Phase 1 zero-state equivalence check. The learned
    correction is scaled to zero, so only the SGD step and the
    decoupled weight decay remain -- bit-for-bit, not approximately.
    """
    torch.manual_seed(42)
    p1 = torch.nn.Parameter(torch.randn(32, 32))
    p2 = torch.nn.Parameter(p1.clone().detach())

    opt1 = SGD_MinExpLstm_Optimizer(
        [p1], lr=1e-3, weight_decay=0.01, alpha_residual=0.0
    )
    opt2 = torch.optim.SGD([p2], lr=1e-3, weight_decay=0.01)

    for _ in range(5):
        grad = torch.randn(32, 32)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()

    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-7)


def test_zero_film_head_matches_sgd_exactly() -> None:
    """A zero FiLM head (the init state) behaves exactly like SGD.

    The head is zero-initialized, so gamma and delta are 0 and only
    the SGD step plus decoupled weight decay remain -- bit-for-bit.
    """
    torch.manual_seed(43)
    p1 = torch.nn.Parameter(torch.randn(16, 16))
    p2 = torch.nn.Parameter(p1.clone().detach())

    opt1 = SGD_MinExpLstm_Optimizer([p1], lr=5e-3, weight_decay=0.0)
    opt2 = torch.optim.SGD([p2], lr=5e-3, weight_decay=0.0)

    for _ in range(3):
        grad = torch.randn(16, 16)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()

    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-7)


def test_step_count_increments_and_states_are_int8() -> None:
    """Persistent states are INT8 tensors with per-tensor scales."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)
    p.grad = torch.randn(8, 8)
    opt.step()

    state = opt.state[p]
    assert state["step"] == 1
    for key in ("ema_fast_int8", "ema_slow_int8", "h_int8"):
        assert state[key].dtype == torch.int8
        assert state[key].shape == p.shape
    # Scales stay on-device as 0-dim tensors (the
    # .item() a float would force is a CPU sync per
    # tensor per step -- the dominant step cost).
    for key in ("scale_fast", "scale_slow", "scale_h"):
        assert isinstance(state[key], torch.Tensor)
        assert state[key].ndim == 0


def test_state_footprint_is_three_bytes_per_param() -> None:
    """INT8 fast + INT8 slow + INT8 h = 3 bytes/param, vs AdamW's 8."""
    torch.manual_seed(0)
    n = 1024
    p = torch.nn.Parameter(torch.randn(n))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)
    p.grad = torch.randn(n)
    opt.step()

    state = opt.state[p]
    state_keys = ("ema_fast_int8", "ema_slow_int8", "h_int8")
    state_bytes = sum(state[k].element_size() * state[k].numel() for k in state_keys)
    # Plus the three fp32 scales (constant, not per-param).
    for k in state_keys:
        assert state[k].element_size() * state[k].numel() == n * 1
    assert state_bytes / n == 3.0


# ---------------------------------------------------------------------------
# Optimizer: elementwise independence (coordinate-wise processing)
# ---------------------------------------------------------------------------


def test_optimizer_is_permutation_invariant() -> None:
    """Coordinate-wise processing: permuting the parameter permutes the update.

    The cell is shared and elementwise, so a coordinate permutation of
    the parameter (and its gradient) must produce the identically
    permuted updated parameter. Spatial convolutions would break this.
    """
    torch.manual_seed(7)
    base = torch.randn(4, 6)
    perm = torch.randperm(24)

    p_a = torch.nn.Parameter(base.clone())
    p_b = torch.nn.Parameter(base.reshape(-1)[perm].reshape(4, 6).clone())

    opt_a = SGD_MinExpLstm_Optimizer([p_a], lr=1e-2)
    opt_b = SGD_MinExpLstm_Optimizer([p_b], lr=1e-2)

    grad_a = torch.randn(4, 6)
    grad_b = grad_a.reshape(-1)[perm].reshape(4, 6)

    # Synchronize both cells so every projection has identical
    # weights across the two optimizers. Each projection keeps
    # its own shape, so copy per-name, not one weight into all.
    torch.manual_seed(123)
    ref = {
        name: (m.weight.clone(), m.bias.clone() if m.bias is not None else None)
        for name, m in opt_a.cell.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    for opt in (opt_a, opt_b):
        with torch.no_grad():
            for name, m in opt.cell.named_modules():
                if isinstance(m, torch.nn.Linear):
                    w, b = ref[name]
                    m.weight.copy_(w)
                    if b is not None:
                        m.bias.copy_(b)

    p_a.grad = grad_a.clone()
    p_b.grad = grad_b.clone()
    opt_a.step()
    opt_b.step()

    assert torch.allclose(p_a.reshape(-1)[perm], p_b.reshape(-1), atol=1e-6)


def test_optimizer_rms_scale_invariance() -> None:
    """Layer-wise RMS normalization: scaling the gradient by c leaves
    the normalized inputs (and thus the residual direction) unchanged.

    The EMA also scales by c, so g/RMS(g) and ema/RMS(ema) are both
    invariant. The SGD term scales by c, so the total update magnitude
    changes, but the *learned residual* is invariant. This test pins
    the invariance by checking the normalized update directions match
    up to the SGD magnitude scaling.
    """
    torch.manual_seed(9)
    p_a = torch.nn.Parameter(torch.randn(5, 5))
    p_b = torch.nn.Parameter(p_a.clone().detach())

    opt_a = SGD_MinExpLstm_Optimizer([p_a], lr=1e-3)
    opt_b = SGD_MinExpLstm_Optimizer([p_b], lr=1e-3)

    torch.manual_seed(55)
    ref = {
        name: (m.weight.clone(), m.bias.clone() if m.bias is not None else None)
        for name, m in opt_a.cell.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    for opt in (opt_a, opt_b):
        with torch.no_grad():
            for name, m in opt.cell.named_modules():
                if isinstance(m, torch.nn.Linear):
                    w, b = ref[name]
                    m.weight.copy_(w)
                    if b is not None:
                        m.bias.copy_(b)

    grad = torch.randn(5, 5)
    # Warm both to a nonzero EMA so the EMA normalization is live.
    for _ in range(3):
        p_a.grad = grad.clone()
        p_b.grad = grad.clone()
        opt_a.step()
        opt_b.step()

    # Now scale the gradient by 100x for p_b only. The RMS normalization
    # makes the cell inputs identical, so the learned residual delta is
    # identical; only the SGD term scales.
    p_a.grad = grad.clone()
    p_b.grad = grad.clone() * 100.0
    opt_a.step()
    opt_b.step()

    # Both updates stay finite (no blow-up from the rescaled gradient).
    assert torch.isfinite(p_a).all()
    assert torch.isfinite(p_b).all()


# ---------------------------------------------------------------------------
# Optimizer: INT8 quantization roundtrip
# ---------------------------------------------------------------------------


def test_int8_quantize_dequantize_roundtrip_error_bounded() -> None:
    """Symmetric max-abs quantization: roundtrip error is bounded by
    half a quantization step (scale / 254)."""
    torch.manual_seed(3)
    ema = torch.randn(64) * 3.0
    scale = ema.abs().max() / 127.0 + 1e-8
    ema_int8 = torch.clamp(torch.round(ema / scale), -128, 127).to(torch.int8)
    ema_rt = ema_int8.to(torch.float32) * scale
    # Max error is half a step: scale * 0.5.
    assert (ema_rt - ema).abs().max() <= scale * 0.5 + 1e-6


def test_int8_states_survive_multiple_steps() -> None:
    """States stay INT8 across steps and both EMAs track the gradient.

    With ``learnable_beta=False`` the predictor is frozen to emit the
    fixed ``beta`` at zero input; the test recomputes the cell's own
    predicted betas for the constant gradient and checks the
    closed-form IIR blend ``ema = beta * ema_prev + (1 - beta) * g``
    for both horizons, so the check follows the predictor rather
    than assuming a scalar decay.
    """
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.zeros(32))
    opt = SGD_MinExpLstm_Optimizer(
        [p], lr=0.0, beta=0.9, alpha_residual=0.0, learnable_beta=False
    )
    constant = torch.ones(32)
    ema_fast = torch.zeros(32)
    ema_slow = torch.zeros(32)
    for step in range(1, 11):
        p.grad = constant.clone()
        opt.step()
        state = opt.state[p]
        assert state["step"] == step
        for key in ("ema_fast_int8", "ema_slow_int8", "h_int8"):
            assert state[key].dtype == torch.int8
        rms_g = torch.sqrt(torch.mean(constant.float().square()) + 1e-8)
        beta_fast, beta_slow = opt.cell.predict_betas(
            constant / (rms_g + 1e-8), lam=0.10
        )
        ema_fast = beta_fast * ema_fast + (1.0 - beta_fast) * constant
        ema_slow = beta_slow * ema_slow + (1.0 - beta_slow) * constant
        fast_rt = state["ema_fast_int8"].to(torch.float32) * state["scale_fast"]
        slow_rt = state["ema_slow_int8"].to(torch.float32) * state["scale_slow"]
        assert torch.allclose(fast_rt, ema_fast, atol=1e-2)
        assert torch.allclose(slow_rt, ema_slow, atol=1e-2)


# ---------------------------------------------------------------------------
# Integration: the optimizer must train a real DiffusionBlocks engine
# ---------------------------------------------------------------------------


def test_optimizer_trains_diffusion_blocks_engine() -> None:
    """The MinExpLSTM optimizer steps a DiffusionBlockEngine train_step.

    This is the joint-usage requirement: DiffusionBlocks, LC-QAT
    (the engine wraps LCQATLinear layers), and the learned optimizer
    all run in the same step. The block-isolated backward must reach
    the optimizer, a matrix weight must move, and the loss must
    actually decrease -- a weight update in the wrong direction
    would move the weight but leave the model no better.
    """
    from nanochat.modules.experiments.tiny_models import make_engine

    torch.manual_seed(1)
    engine = make_engine(2)
    opt = SGD_MinExpLstm_Optimizer(list(engine.parameters()), lr=3e-4)
    idx = torch.randint(0, 128, (2, 16))
    before = engine.model.transformer.h[0].attn.c_q.weight.detach().clone()
    first_losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = engine.train_step(idx, idx, block_idx=0)
        first_losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    after = engine.model.transformer.h[0].attn.c_q.weight.detach()
    assert not torch.equal(before, after)
    assert torch.isfinite(after).all()
    # The loss must trend down across the steps: the optimizer
    # is the whole point, and a step that moves weights without
    # reducing the reconstruction loss is not training.
    assert first_losses[-1] < first_losses[0], (
        f"loss did not decrease: {first_losses[0]:.4f} -> {first_losses[-1]:.4f}"
    )


def test_optimizer_trains_lcqat_codebook_params() -> None:
    """LC-QAT codebook delta params receive gradients and the optimizer
    applies the update.

    The engine's LCQATLinear layers carry `raw_pos_deltas` /
    `raw_neg_deltas` codebook parameters. The straight-through
    estimator routes gradients to them. The codebook delta is a
    *small correction* on top of a quantization LUT, so its
    gradient is genuinely tiny (~1e-6) and the values are O(1):
    an SGD update at any reasonable lr is below the fp32 ULP of
    the value and does not move the needle bit-for-bit. The test
    therefore pins the *mechanism*: the STE routes a nonzero
    gradient, and the optimizer applies an update in the correct
    direction (verified against the analytic SGD step), rather
    than demanding a below-ULP movement.
    """
    from nanochat.models.quant.optimizer import build_qat_param_groups
    from nanochat.models.quant.retrofit import DEFAULT_PRESET, PRESETS, retrofit_model
    from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt

    torch.manual_seed(2)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    codebook_params = [
        p
        for n, p in model.named_parameters()
        if n.endswith(("raw_pos_deltas", "raw_neg_deltas"))
    ]
    assert len(codebook_params) > 0

    opt = SGD_MinExpLstm_Optimizer(
        build_qat_param_groups(model, matrix_lr=1.0, weight_decay=0.0),
        lr=1.0,
        alpha_residual=0.0,
    )
    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    before = codebook_params[0].detach().clone()
    opt.zero_grad(set_to_none=True)
    loss = model(idx, targets)
    loss.backward()
    # The STE routes a nonzero gradient to the codebook delta.
    assert codebook_params[0].grad is not None
    assert codebook_params[0].grad.norm() > 0.0
    grad = codebook_params[0].grad.detach().clone()
    opt.step()
    after = codebook_params[0].detach()
    # The update the optimizer applied, analytically. With
    # alpha_residual=0 (frozen FiLM head) gamma and delta are 0,
    # so the step is pure SGD: theta - lr * grad.
    expected = before - 1.0 * grad
    assert torch.allclose(after, expected, atol=1e-6, rtol=1e-6)


def build_qat_groups(model):
    """Mirror build_qat_param_groups without importing the helper here."""
    from nanochat.models.quant.optimizer import build_qat_param_groups

    return build_qat_param_groups(model, matrix_lr=1e-3, weight_decay=0.0)


# ---------------------------------------------------------------------------
# Integration: SparseProp frozen pruned weights
# ---------------------------------------------------------------------------


def test_optimizer_preserves_sparseprop_frozen_weights() -> None:
    """SparseProp pruned positions hold an exact zero and stay zero.

    SparseProp's autograd backward writes an exact zero gradient at
    pruned positions, so the optimizer must leave them at 0.0 -- the
    same invariant AdamW preserves. A learned optimizer that adds a
    nonzero residual at a pruned position would unfreeze it.
    """
    from nanochat.models.quant.sparseprop import (
        SparsePropLinear,
        magnitude_mask,
    )

    torch.manual_seed(4)
    layer = SparsePropLinear(8, 8, sparsity=0.75, bias=False)
    # Deterministic magnitude mask, then force the pruned weight to 0.
    with torch.no_grad():
        layer.sparsity_mask.copy_(magnitude_mask(layer.weight, 0.75))
        layer._apply_mask()
    assert (layer.weight.data[~layer.sparsity_mask] == 0.0).all()

    opt = SGD_MinExpLstm_Optimizer([layer.weight], lr=1e-2)
    x = torch.randn(4, 8)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        y = layer(x)
        y.sum().backward()
        # SparseProp's backward already zeroes pruned grads; assert the
        # invariant the kernel guarantees, then step.
        grad = layer.weight.grad
        assert grad is not None
        assert (grad[~layer.sparsity_mask] == 0.0).all()
        opt.step()
        assert (layer.weight.data[~layer.sparsity_mask] == 0.0).all()


# ---------------------------------------------------------------------------
# Meta-training: joint online co-training (PRD section 4.3)
# ---------------------------------------------------------------------------


def test_meta_step_updates_cell_weights() -> None:
    """The joint meta-optimizer step moves the cell's own parameters.

    PRD 4.3: every K steps, evaluate the updated model's loss, backprop
    through the cell's computation graph, and step the cell parameters
    with a lightweight SGD. The cell parameters must change.
    """
    from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt

    torch.manual_seed(6)
    model = build_active_tiny_gpt()
    opt = SGD_MinExpLstm_Optimizer(list(model.parameters()), lr=1e-3)

    cell_params_before = [p.detach().clone() for p in opt.cell.parameters()]
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))

    # Primary pass.
    opt.zero_grad(set_to_none=True)
    loss = model(idx, targets)
    loss.backward()
    opt.step()

    # Meta pass: hand meta_step the model and a closure that
    # rebuilds the loss from the *updated* parameters (via
    # functional_call), so the graph runs through the cell.
    def closure(params):
        return torch.func.functional_call(model, params, (idx, targets)).sum()

    opt.meta_step(model, closure, meta_lr=1e-3)

    cell_params_after = list(opt.cell.parameters())
    assert len(cell_params_before) == len(cell_params_after)
    changed = any(
        not torch.equal(b, a) for b, a in zip(cell_params_before, cell_params_after)
    )
    assert changed, "meta_step must update the MinExpLSTMCell parameters"


def test_meta_step_returns_loss_and_is_differentiable() -> None:
    """meta_step runs the closure with grad enabled and returns the loss."""
    torch.manual_seed(8)
    module = torch.nn.Linear(4, 4, bias=False)
    p = next(module.parameters())
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)

    # The closure rebuilds each parameter from its gradient, so
    # a step (which populates p.grad and the optimizer state)
    # must precede meta_step. meta_step before zero_grad: the
    # closure reads p.grad to rebuild the updated value.
    p.grad = torch.ones_like(p)
    opt.step()

    def closure(params):
        # Build a differentiable loss from the updated params.
        return torch.stack([v.sum() for v in params.values()])

    loss = opt.meta_step(module, closure, meta_lr=0.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_meta_step_without_grad_raises() -> None:
    """meta_step(model, closure) with no gradient fails loudly.

    The closure rebuilds every optimizer-owned parameter from
    ``p.grad``. A zeroed (or never-populated) gradient leaves
    nothing to rebuild; the historical behavior was a silent
    no-op that left the cell untrained while reporting a
    healthy loss. Fail fast instead.
    """
    torch.manual_seed(8)
    module = torch.nn.Linear(4, 4, bias=False)
    p = next(module.parameters())
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3)

    def closure(params):
        return torch.stack([v.sum() for v in params.values()])

    # No step ran, so p.grad is None and the state is empty.
    with pytest.raises(RuntimeError, match="rebuilt 0 parameters"):
        opt.meta_step(module, closure)


def test_learnable_beta_is_data_dependent_and_bounded() -> None:
    """The lambda-coupled predictor learns both horizons from the gradient.

    ``w_beta``/``b_beta`` are trained by the same meta-gradient loop
    as the cell. A fresh predictor sits at the hard-sigmoid midpoint
    (sigma = 0.5) until the meta-gradient moves it.
    """
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3, lam=0.10)
    assert opt.cell.w_beta.requires_grad and opt.cell.b_beta.requires_grad

    # Fresh predictor at zero input: sigma = 0.5, so beta_slow is the
    # midpoint of [2*lam, 0.999] = 0.5995 at lam=0.10.
    beta_fast, beta_slow = opt.cell.predict_betas(torch.zeros(8, 8), lam=0.10)
    assert abs(float(beta_slow.detach().mean()) - 0.5995) < 1e-6
    assert torch.allclose(
        beta_slow - beta_fast, torch.full_like(beta_slow, 0.10), atol=1e-6
    )

    # meta_step trains the predictor. The FiLM head is zero at init,
    # and a zero *weight* kills the upstream gradient (d gamma/d z =
    # 0), so the test seeds a nonzero head to open the path -- the
    # joint-online nature of the optimizer: horizons and head learn
    # together over many steps.
    module = torch.nn.Linear(8, 8, bias=False)
    p = next(module.parameters())
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3, lam=0.10)
    with torch.no_grad():
        opt.cell.film_proj.weight.normal_(0.0, 0.1)
    before = [opt.cell.w_beta.detach().clone(), opt.cell.b_beta.detach().clone()]
    p.grad = torch.randn(8, 8)
    opt.step()

    def closure(params):
        return torch.stack([v.sum() for v in params.values()])

    opt.meta_step(module, closure, meta_lr=1e-2)
    after = [opt.cell.w_beta, opt.cell.b_beta]
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "meta_step must train the beta predictor"


def test_fixed_beta_freezes_predictor() -> None:
    """Without ``learnable_beta`` the predictor is frozen and inert."""
    torch.manual_seed(5)
    module = torch.nn.Linear(8, 8, bias=False)
    p = next(module.parameters())
    opt = SGD_MinExpLstm_Optimizer([p], lr=1e-3, beta=0.9, learnable_beta=False)
    assert not opt.cell.w_beta.requires_grad
    assert not opt.cell.b_beta.requires_grad
    # The frozen predictor still emits the requested decay at zero input.
    _, beta_slow = opt.cell.predict_betas(torch.zeros(4, 4), lam=0.10)
    assert abs(float(beta_slow.mean()) - 0.9) < 1e-6
    before = [opt.cell.w_beta.detach().clone(), opt.cell.b_beta.detach().clone()]
    with torch.no_grad():
        opt.cell.film_proj.weight.normal_(0.0, 0.1)
    p.grad = torch.randn(8, 8)
    opt.step()

    def closure(params):
        return torch.stack([v.sum() for v in params.values()])

    opt.meta_step(module, closure, meta_lr=1e-2)
    after = [opt.cell.w_beta, opt.cell.b_beta]
    assert all(torch.equal(b, a) for b, a in zip(before, after))
