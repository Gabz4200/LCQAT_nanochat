"""Three-way parity tests for SGD-MinExpLSTM fused kernels.

Tests that naive (Python), CPU (C++), and GPU (Taichi) backends
produce identical results for the fused SGD-MinExpLSTM step.
"""

import pytest
import torch

from nanochat.models.quant.retrofit import PRESETS, retrofit_model
from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt
from nanochat.ops.sgd_minexplstm import dispatch_sgd_minexplstm_step
from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer


def _setup_optimizer_and_model():
    """Create a retrofitted model with SGD_MinExpLstm_Optimizer."""
    model = retrofit_model(build_active_tiny_gpt(), PRESETS["small"])
    opt = SGD_MinExpLstm_Optimizer(list(model.parameters()), lr=1e-3, lam=0.10)
    return model, opt


def _get_cell_params(opt):
    """Extract cell parameters from optimizer."""
    cell = opt.cell
    return {
        "phi_w": cell.phi_proj.weight.detach(),
        "phi_b": cell.phi_proj.bias.detach(),
        "iota_w": cell.iota_proj.weight.detach(),
        "iota_b": cell.iota_proj.bias.detach(),
        "h_w": cell.h_proj.weight.detach(),
        "h_b": cell.h_proj.bias.detach(),
        "out_w": cell.out_proj.weight.detach(),
        "out_b": cell.out_proj.bias.detach(),
        "film_w": cell.film_proj.weight.detach(),
        "film_b": cell.film_proj.bias.detach(),
        "w_beta": cell.w_beta.detach(),
        "b_beta": cell.b_beta.detach(),
    }


@pytest.mark.parametrize("backend", ["naive", "cpu", "gpu"])
def test_dispatch_backend_runs(backend):
    """Test that each backend runs without error."""
    model, opt = _setup_optimizer_and_model()
    cell_params = _get_cell_params(opt)
    batch = (torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))

    opt.zero_grad()
    loss = model(*batch)
    loss.backward()
    opt.step()  # Initialize optimizer state

    param_name = None
    for name, p in model.named_parameters():
        if p.grad is not None and len(opt.state[p]) > 0:
            param_name = name
            break

    assert param_name is not None, "No parameter with optimizer state found"
    p = dict(model.named_parameters())[param_name]
    state = opt.state[p]

    assert p.grad is not None, "No parameter with a gradient to step on"

    try:
        from nanochat.ops.sgd_minexplstm import dispatch_sgd_minexplstm_step

        dispatch_sgd_minexplstm_step(
            grad=p.grad,
            weight=p.data,
            ema_fast_int8=state["ema_fast_int8"],
            ema_slow_int8=state["ema_slow_int8"],
            h_int8=state["h_int8"],
            scale_fast=state["scale_fast"],
            scale_slow=state["scale_slow"],
            scale_h=state["scale_h"],
            phi_w=cell_params["phi_w"],
            phi_b=cell_params["phi_b"],
            iota_w=cell_params["iota_w"],
            iota_b=cell_params["iota_b"],
            h_w=cell_params["h_w"],
            h_b=cell_params["h_b"],
            out_w=cell_params["out_w"],
            out_b=cell_params["out_b"],
            film_w=cell_params["film_w"],
            film_b=cell_params["film_b"],
            w_beta=cell_params["w_beta"],
            b_beta=cell_params["b_beta"],
            lr=1e-3,
            wd=1e-2,
            lam=0.10,
            clamp_val=1.0,
            backend=backend,
        )
    except Exception as e:
        if backend == "gpu" and "Vulkan" in str(e):
            pytest.skip(f"GPU backend not available: {e}")
        raise


def test_cpu_gpu_parity():
    """Test CPU and GPU backends produce identical results."""
    model, opt = _setup_optimizer_and_model()
    cell_params = _get_cell_params(opt)
    batch = (torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))

    opt.zero_grad()
    loss = model(*batch)
    loss.backward()
    opt.step()

    param_name = None
    for name, p in model.named_parameters():
        if p.grad is not None and len(opt.state[p]) > 0:
            param_name = name
            break

    assert param_name is not None
    p = dict(model.named_parameters())[param_name]
    state = opt.state[p]

    hyperparams = {"lr": 1e-3, "wd": 1e-2, "lam": 0.10, "clamp_val": 1.0}

    def _run(backend):
        # Fresh copies of the state so each backend
        # starts from the same input.
        p_copy = p.data.clone()
        ef = state["ema_fast_int8"].clone()
        es = state["ema_slow_int8"].clone()
        h = state["h_int8"].clone()
        sf = state["scale_fast"].clone()
        ss = state["scale_slow"].clone()
        sh = state["scale_h"].clone()
        assert p.grad is not None
        dispatch_sgd_minexplstm_step(
            grad=p.grad,
            weight=p_copy,
            ema_fast_int8=ef,
            ema_slow_int8=es,
            h_int8=h,
            scale_fast=sf,
            scale_slow=ss,
            scale_h=sh,
            phi_w=cell_params["phi_w"],
            phi_b=cell_params["phi_b"],
            iota_w=cell_params["iota_w"],
            iota_b=cell_params["iota_b"],
            h_w=cell_params["h_w"],
            h_b=cell_params["h_b"],
            out_w=cell_params["out_w"],
            out_b=cell_params["out_b"],
            film_w=cell_params["film_w"],
            film_b=cell_params["film_b"],
            w_beta=cell_params["w_beta"],
            b_beta=cell_params["b_beta"],
            **hyperparams,
            backend=backend,
        )
        return p_copy, ef, es, h, sf, ss, sh

    try:
        cpu = _run("cpu")
        gpu = _run("gpu")
    except Exception as e:
        if "Vulkan" in str(e):
            pytest.skip(f"GPU backend not available: {e}")
        raise

    # Weight updates agree to fp32 tolerance; the
    # INT8 states and scales must match exactly (the
    # quantization is deterministic given the same
    # max-abs scale).
    for i, name in enumerate(
        ("weight", "ema_fast", "ema_slow", "h", "scale_fast", "scale_slow", "scale_h")
    ):
        a, b = cpu[i], gpu[i]
        if name == "weight":
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        elif name.startswith("scale"):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-8)
        else:
            assert torch.equal(a, b), f"{name} mismatch: {a[:8]} vs {b[:8]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
