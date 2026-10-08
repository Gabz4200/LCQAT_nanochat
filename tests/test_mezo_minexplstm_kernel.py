"""Kernel parity and behavior tests for the MeZO-MinExpLSTM optimizer.

Pins the three fused backends (naive / CPU C++ / GPU Taichi) to
identical output on the same inputs, the same way
``tests/test_minexplstm_kernel.py`` does for the SGD stack:
three-way parity ``naive == cpu == gpu`` plus the optimizer-level
contracts (seed-replay determinism, INT8 state quantization,
coordinate-wise clipping, and the division-free recurrence).
"""

from __future__ import annotations

import pytest
import torch

from nanochat.ops.mezo_minexplstm import (
    _mezo_minexplstm_step_naive,
    dispatch_mezo_minexplstm_step,
)
from nanochat.training.mezo_minexplstm import (
    MeZo_MinExpLstm_Optimizer,
    MeZoMinExpLSTMCellFiLM4D,
    _derive_seed,
)


def _rand_cell(seed: int = 0) -> MeZoMinExpLSTMCellFiLM4D:
    g = torch.Generator().manual_seed(seed)
    cell = MeZoMinExpLSTMCellFiLM4D(K=4)
    for p in cell.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.1
    return cell


def _step_fused(
    backend: str,
    fir: torch.Tensor,
    mezo: torch.Tensor,
    weight: torch.Tensor,
    h_int8: torch.Tensor,
    scale_h: torch.Tensor,
    cell: MeZoMinExpLSTMCellFiLM4D,
    lr: float = 1e-3,
    wd: float = 1e-2,
    clamp_val: float = 1.0,
    clip_bound: float = 1e-3,
) -> None:
    """Run one fused MeZO step through the selected backend.

    The fused kernels operate on flat [N] layouts (the same
    contract the SGD-MinExpLSTM kernels use), so 2D parameter
    tensors are flattened here and reshaped back by the caller
    only for the parity comparison.
    """
    if backend == "naive":
        _mezo_minexplstm_step_naive(
            fir,
            mezo,
            weight,
            h_int8,
            scale_h,
            cell.phi_proj.weight.reshape(-1),
            cell.phi_proj.bias,
            cell.iota_proj.weight.reshape(-1),
            cell.iota_proj.bias,
            cell.h_proj.weight.reshape(-1),
            cell.h_proj.bias,
            cell.out_proj.weight.reshape(-1),
            cell.out_proj.bias,
            cell.film_proj.weight.reshape(-1),
            cell.film_proj.bias,
            lr,
            wd,
            clamp_val,
            clip_bound,
        )
    else:
        dispatch_mezo_minexplstm_step(
            fir.reshape(-1),
            mezo.reshape(-1),
            weight.reshape(-1),
            h_int8.reshape(-1),
            scale_h,
            cell.phi_proj.weight.reshape(-1),
            cell.phi_proj.bias,
            cell.iota_proj.weight.reshape(-1),
            cell.iota_proj.bias,
            cell.h_proj.weight.reshape(-1),
            cell.h_proj.bias,
            cell.out_proj.weight.reshape(-1),
            cell.out_proj.bias,
            cell.film_proj.weight.reshape(-1),
            cell.film_proj.bias,
            lr,
            wd,
            clamp_val,
            clip_bound,
            backend=backend,  # type: ignore[arg-type]
        )


def _clone_step_inputs(
    n: int,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, ...]:
    """Build identical (fir, mezo, weight, h_int8, scale_h) sets."""
    g = torch.Generator().manual_seed(seed)
    fir = torch.randn(n, generator=g, device=device, dtype=torch.float32)
    mezo = torch.randn(n, generator=g, device=device, dtype=torch.float32)
    weight = torch.randn(n, generator=g, device=device, dtype=torch.float32)
    h_int8 = torch.randint(
        -128, 128, (n,), generator=g, device=device, dtype=torch.int8
    )
    scale_h = torch.tensor(0.05, device=device, dtype=torch.float32)
    return fir, mezo, weight, h_int8, scale_h


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_when_mezo_kernel_runs_then_matches_naive(backend: str) -> None:
    """Three-way parity: naive == cpu == gpu on identical inputs."""
    n = 256
    cell = _rand_cell(42)

    # Naive reference run.
    fir_n, mezo_n, w_n, h_n, s_n = _clone_step_inputs(n, seed=1)
    _step_fused("naive", fir_n, mezo_n, w_n, h_n, s_n, cell)

    # Fused backend run on fresh copies.
    fir_b, mezo_b, w_b, h_b, s_b = _clone_step_inputs(n, seed=1)
    _step_fused(backend, fir_b, mezo_b, w_b, h_b, s_b, cell)

    torch.testing.assert_close(w_b, w_n, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(h_b.to(torch.int32), h_n.to(torch.int32))
    torch.testing.assert_close(s_b, s_n, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_when_mezo_kernel_steps_repeat_then_stays_exact(backend: str) -> None:
    """Multi-step parity accumulates no drift (weight-decay form).

    A wrong decoupled-weight-decay form (lr^2 * wd instead of
    lr * wd) passes a single loose-tolerance step but drifts
    linearly with step count. Feeding the same step inputs
    repeatedly and comparing to the naive reference at tight
    tolerance pins the exact update formula.
    """
    n = 256
    cell = _rand_cell(42)
    lr, wd, clamp_val, clip_bound = 1e-2, 1e-2, 1.0, 1e-3

    # Naive reference, 20 identical steps.
    fir_n, mezo_n, w_n, h_n, s_n = _clone_step_inputs(n, seed=1)
    for _ in range(20):
        _step_fused(
            "naive",
            fir_n,
            mezo_n,
            w_n,
            h_n,
            s_n,
            cell,
            lr=lr,
            wd=wd,
            clamp_val=clamp_val,
            clip_bound=clip_bound,
        )

    # Fused backend, 20 identical steps.
    fir_b, mezo_b, w_b, h_b, s_b = _clone_step_inputs(n, seed=1)
    for _ in range(20):
        _step_fused(
            backend,
            fir_b,
            mezo_b,
            w_b,
            h_b,
            s_b,
            cell,
            lr=lr,
            wd=wd,
            clamp_val=clamp_val,
            clip_bound=clip_bound,
        )

    torch.testing.assert_close(w_b, w_n, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(h_b.to(torch.int32), h_n.to(torch.int32))
    torch.testing.assert_close(s_b, s_n, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_when_mezo_kernel_runs_on_batched_params_then_matches_naive(
    backend: str,
) -> None:
    """Parity must hold for a 2D parameter (row-major flattened layout).

    The fused kernels consume flat [N] views of the parameter, so
    the test passes flattened views of the same underlying storage
    and compares the reshaped results.
    """
    rows, cols = 16, 32
    n = rows * cols
    cell = _rand_cell(7)

    fir_n, mezo_n, w_n_flat, h_n_flat, s_n = _clone_step_inputs(n, seed=3)
    w_n = w_n_flat.reshape(rows, cols)
    h_n = h_n_flat.reshape(rows, cols)
    _step_fused("naive", fir_n, mezo_n, w_n, h_n, s_n, cell)

    fir_b, mezo_b, w_b_flat, h_b_flat, s_b = _clone_step_inputs(n, seed=3)
    w_b = w_b_flat.reshape(rows, cols)
    h_b = h_b_flat.reshape(rows, cols)
    _step_fused(backend, fir_b, mezo_b, w_b, h_b, s_b, cell)

    torch.testing.assert_close(w_b, w_n, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h_b.to(torch.int32), h_n.to(torch.int32))


def test_when_derive_seed_called_then_deterministic_and_well_separated() -> None:
    """Per-parameter seed derivation is deterministic and separates indices."""
    s0 = _derive_seed(123, 0)
    s1 = _derive_seed(123, 1)
    s0_again = _derive_seed(123, 0)
    assert s0 == s0_again
    assert s0 != s1
    assert all(0 <= s < 2**32 for s in (s0, s1))


def test_when_optimizer_steps_then_seed_replay_is_deterministic() -> None:
    """Same seed + same losses -> identical parameter update (MeZO replay)."""
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 16))]
    params_b = [torch.nn.Parameter(params_a[0].data.clone())]

    opt_a = MeZo_MinExpLstm_Optimizer(params_a, backend="naive")
    opt_b = MeZo_MinExpLstm_Optimizer(params_b, backend="naive")

    for opt in (opt_a, opt_b):
        opt.stage_mezo_estimate(123, torch.tensor(1.5), torch.tensor(1.0))
        opt.step()

    torch.testing.assert_close(params_a[0].data, params_b[0].data)


def test_when_optimizer_steps_then_int8_state_quantized() -> None:
    """The persistent hidden state is INT8 with a per-tensor scale."""
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = MeZo_MinExpLstm_Optimizer([p], backend="naive")
    opt.stage_mezo_estimate(123, torch.tensor(1.5), torch.tensor(1.0))
    opt.step()

    state = opt.state[p]
    assert state["h_int8"].dtype == torch.int8
    assert state["h_int8"].shape == p.shape
    assert state["scale_h"].numel() == 1
    # Dequantized state is finite and bounded near the cell's range.
    h = state["h_int8"].to(torch.float32) * state["scale_h"]
    assert torch.isfinite(h).all()


def test_when_clip_bound_applied_then_update_bounded() -> None:
    """Coordinate-wise clipping bounds the update magnitude (PRD 2.5)."""
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = MeZo_MinExpLstm_Optimizer(
        [p],
        lr=1.0,  # large lr to exercise the clip
        epsilon=1e-3,
        a_base=1.5,
        a_min=0.1,
        delta_q=0.0625,
        backend="naive",
    )
    before = p.data.clone()
    opt.stage_mezo_estimate(123, torch.tensor(10.0), torch.tensor(-10.0))
    opt.step()

    # clip_bound = a_t * delta_q = 1.5 * 0.0625 = 0.09375 at lr == lr_max.
    # The update is lr * clipped, so |delta| <= lr * clip_bound.
    max_delta = (p.data - before).abs().max().item()
    assert max_delta <= 1.0 * 0.09375 + 1e-6


def test_when_meta_step_tensor_then_trains_cell() -> None:
    """meta_step(loss) backprops a loss wired to the cell and updates it."""
    cell = _rand_cell(11)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = MeZo_MinExpLstm_Optimizer([p], backend="naive")
    opt.cell = cell

    # Build a differentiable loss through the cell: a quadratic
    # target so every parameter of the cell has a nonzero gradient.
    x = torch.randn(16, 3)
    h = torch.randn(16, 1)
    gamma, beta, h_next = cell(x, h)
    target = torch.randn_like(gamma)
    loss = ((gamma - target) ** 2).mean() + ((h_next - target) ** 2).mean()

    before = [q.detach().clone() for q in cell.parameters()]
    opt.meta_step(loss)
    after = list(cell.parameters())

    changed = any(not torch.equal(b, a.detach()) for b, a in zip(before, after))
    assert changed


def test_when_meta_step_closure_then_trains_cell() -> None:
    """meta_step(model, closure) rebuilds the loss from updated params."""
    torch.manual_seed(1)
    model = torch.nn.Linear(4, 4)
    opt = MeZo_MinExpLstm_Optimizer(list(model.parameters()), backend="naive")

    # A step first: the closure rebuilds each parameter
    # from its step state (FIR history, hidden state), so
    # seed_history/mu_history and the state must be
    # populated before meta_step.
    x = torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))
    opt.perturb(1, scaling=1.0)
    loss_pos = torch.nn.functional.cross_entropy(model(x), y)
    opt.perturb(1, scaling=-2.0)
    loss_neg = torch.nn.functional.cross_entropy(model(x), y)
    opt.perturb(1, scaling=1.0)
    opt.stage_mezo_estimate(1, loss_pos, loss_neg)
    opt.step()

    target = torch.randn(4, 4)

    def closure(params: dict) -> torch.Tensor:
        w = params["weight"]
        return torch.nn.functional.mse_loss(w, target)

    cell_before = [q.detach().clone() for q in opt.cell.parameters()]
    loss = opt.meta_step(model, closure)
    # The closure rebuilt the loss from updated params through the
    # cell, so the returned loss is differentiable (the optimizer
    # raises TypeError otherwise). A vacuous `or True` here would
    # let a silent no-op pass.
    assert loss.requires_grad is True
    assert any(
        not torch.equal(b, q.detach())
        for b, q in zip(cell_before, opt.cell.parameters())
    ), "meta_step closure must train the cell parameters"


def test_when_no_staged_estimate_then_step_raises() -> None:
    """step() without stage_mezo_estimate fails loudly (fail-fast)."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = MeZo_MinExpLstm_Optimizer([p], backend="naive")
    with pytest.raises(RuntimeError, match="staged MeZO"):
        opt.step()


def test_when_invalid_epsilon_then_init_raises() -> None:
    """Invalid hyperparameters fail at construction, not mid-training."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="epsilon"):
        MeZo_MinExpLstm_Optimizer([p], epsilon=0.0)
    with pytest.raises(ValueError, match="K"):
        MeZo_MinExpLstm_Optimizer([p], K=0)


def test_when_unknown_backend_then_dispatch_raises() -> None:
    """Unknown backends fail loudly rather than degrading to naive."""
    n = 64
    fir, mezo, w, h, s = _clone_step_inputs(n, seed=5)
    cell = _rand_cell(3)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_mezo_minexplstm_step(
            fir,
            mezo,
            w,
            h,
            s,
            cell.phi_proj.weight.reshape(-1),
            cell.phi_proj.bias,
            cell.iota_proj.weight.reshape(-1),
            cell.iota_proj.bias,
            cell.h_proj.weight.reshape(-1),
            cell.h_proj.bias,
            cell.out_proj.weight.reshape(-1),
            cell.out_proj.bias,
            cell.film_proj.weight.reshape(-1),
            cell.film_proj.bias,
            1e-3,
            1e-2,
            1.0,
            1e-3,
            backend="tpu",  # type: ignore[arg-type]
        )


def test_when_perturb_roundtrip_then_params_restored() -> None:
    """MeZO's three perturb phases (+1, -2, +1) restore the baseline."""
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = MeZo_MinExpLstm_Optimizer([p], backend="naive")
    before = p.data.clone()
    opt.perturb(123, scaling=1.0)
    opt.perturb(123, scaling=-2.0)
    opt.perturb(123, scaling=1.0)
    torch.testing.assert_close(p.data, before)


def test_when_meta_step_closure_then_conv_w_receives_gradient() -> None:
    """The FIR kernel (conv_w) is on the closure's gradient path.

    The closure rebuilds the FIR signal differentiably
    (``m_fir = m_fir + z_j * c_j`` with ``c_j = conv_w[j]
    * mu``), so ``conv_w`` must receive a nonzero gradient.
    Its magnitude is far below ``film_proj``'s (the gradient
    traverses the log-domain cell, the ReLU latent, and the
    RMS-normalized FIR input), but a zero gradient would mean
    the FIR filter is frozen and the MeZO-specific
    contribution is inert.
    """
    torch.manual_seed(3)
    model = torch.nn.Linear(32, 32)
    opt = MeZo_MinExpLstm_Optimizer(list(model.parameters()), backend="naive")

    # A few steps to populate the seed/mu history the FIR
    # filter reads.
    x = torch.randn(8, 32)
    y = torch.randint(0, 32, (8,))
    for seed in (1, 2, 3):
        opt.perturb(seed, scaling=1.0)
        loss_pos = torch.nn.functional.cross_entropy(model(x), y)
        opt.perturb(seed, scaling=-2.0)
        loss_neg = torch.nn.functional.cross_entropy(model(x), y)
        opt.perturb(seed, scaling=1.0)
        opt.stage_mezo_estimate(seed, loss_pos, loss_neg)
        opt.step()

    # The FiLM head is zero-initialized (the step-0
    # safeguard), so the first meta-step leaves the
    # FIR kernel with a zero gradient: gamma's gradient
    # to fir_norm flows through film_proj, which sits at
    # its zero fixed point. Two meta-steps -- the first
    # moves the FiLM head off zero, the second gives
    # conv_w a nonzero gradient.
    for _ in range(2):
        for q in opt.cell.parameters():
            q.grad = None

        def closure(params: dict) -> torch.Tensor:
            out = torch.func.functional_call(model, params, (x,))
            return torch.nn.functional.cross_entropy(out, y)

        opt.meta_step(model, closure)

    conv_w_grad = opt.cell.conv_w.grad
    assert conv_w_grad is not None, "conv_w must be on the closure graph"
    assert float(conv_w_grad.abs().sum()) > 0.0, (
        "conv_w must receive a nonzero gradient through the FIR rebuild"
    )
    film_grad = opt.cell.film_proj.weight.grad
    assert film_grad is not None
    # The FIR kernel's gradient is nonzero but smaller than
    # the FiLM head's (the chain traverses the cell); pin
    # the ordering so a severed FIR path (exactly zero) is
    # distinguishable from a healthy small one.
    assert float(conv_w_grad.abs().sum()) < float(film_grad.abs().sum())
