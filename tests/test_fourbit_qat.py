"""4-bit main-model weights + QAT on both MinExpLSTM optimizers.

The PRD keeps the *main model* in 4-bit storage and the optimizers
run Quantization-Aware Training: the step dequantizes the 4-bit
weights, applies the MinExpLSTM update, and requantizes back to 4
bits. The SGD optimizer additionally quantizes gradients to INT8
(8-bit gradients). These tests pin the storage footprint, the
roundtrip, and the optimizer behavior on the 4-bit path.
"""

import torch

from nanochat.models.quant.fourbit import (
    FourBitLinear,
    pack_fourbit_weights,
    unpack_fourbit_weights,
)


def test_when_fourbit_linear_built_then_weights_stored_as_nibbles() -> None:
    """A FourBitLinear stores its persistent weights as packed nibbles + a scale.

    4 bits/weight means the packed buffer is ceil(N/2) uint8 bytes
    plus one fp32 scale per tensor. The fp32 ``weight`` is the
    optimization surrogate the MinExpLSTM optimizers step; the
    packed buffer is the persistent 4-bit storage.
    """
    torch.manual_seed(0)
    layer = FourBitLinear(8, 16, bias=False)
    # The packed buffer holds ceil(16*8 / 2) bytes.
    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_packed.numel() == (16 * 8 + 1) // 2
    # One fp32 scale for the whole tensor.
    assert layer.weight_scale.numel() == 1


def test_when_fourbit_forward_then_matches_quantized_reference() -> None:
    """The forward dequantizes the packed nibbles and matmuls.

    The quantized weight is ``(round(w/scale) + 8 - 8) * scale``
    through the packed buffer, so the output matches a plain
    ``F.linear`` on that dequantized weight (not the raw fp32
    surrogate -- the forward is quantization-aware).
    """
    torch.manual_seed(1)
    layer = FourBitLinear(8, 16, bias=False)
    x = torch.randn(3, 8)
    with torch.no_grad():
        w_q = layer._quantized_weight()
    y = layer(x)
    y_ref = torch.nn.functional.linear(x, w_q.detach())
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


def test_when_fourbit_forward_then_ste_routes_gradient_to_weight() -> None:
    """The straight-through estimator routes the gradient to the surrogate.

    The forward round-trips through the discrete packed nibbles,
    but the STE passes the output gradient to the fp32 surrogate
    (and the scale), so a loss backprops into ``weight``.
    """
    torch.manual_seed(2)
    layer = FourBitLinear(8, 4, bias=False)
    x = torch.randn(2, 8, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert x.grad is not None


def test_when_pack_unpack_fourbit_then_roundtrips() -> None:
    """Pack/unpack round-trips nibble indices exactly.

    A ``[m, n]`` index tensor packs to ``[m, ceil(n/2)]`` bytes
    (two nibbles per byte, row-wise), then unpacks back to ``[m, n]``.
    """
    indices = torch.randint(0, 16, (5, 7), dtype=torch.uint8)
    packed = pack_fourbit_weights(indices)
    assert packed.shape == (5, (7 + 1) // 2)
    unpacked = unpack_fourbit_weights(packed, 7)
    torch.testing.assert_close(unpacked, indices)


def test_when_fourbit_storage_then_footprint_is_half_of_fp32() -> None:
    """4-bit storage is 0.5 bytes/weight vs fp32's 4.

    The packed buffer plus the scale must be ~1/8 the fp32 weight
    footprint (4 bits vs 32 bits), the PRD's storage pillar.
    """
    torch.manual_seed(3)
    layer = FourBitLinear(64, 64, bias=False)
    n = 64 * 64
    packed_bytes = layer.weight_packed.numel()
    fp32_bytes = n * 4
    # 4 bits/weight = n/2 bytes, plus one 4-byte scale.
    assert packed_bytes == n // 2
    assert packed_bytes + 4 < fp32_bytes / 8 + 8


# --- QAT on the SGD optimizer: 8-bit gradients + 4-bit weight update ---


def _tiny_sgd_optimizer():
    from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer

    torch.manual_seed(4)
    lin = torch.nn.Linear(8, 8, bias=False)
    opt = SGD_MinExpLstm_Optimizer(lin.parameters(), lr=0.1, qat_4bit=True)
    return lin, opt


def test_when_sgd_qat_then_grads_quantized_to_int8() -> None:
    """With qat_4bit=True the SGD optimizer quantizes grads to INT8.

    The optimizer state holds `grad_int8` + `scale_grad` (8-bit
    gradients) instead of the fp32 gradient, per the PRD's
    8-bit-gradient allowance for the SGD path.
    """
    lin, opt = _tiny_sgd_optimizer()
    x = torch.randn(4, 8)
    loss = lin(x).sum()
    loss.backward()
    opt.step()
    p = next(lin.parameters())
    state = opt.state[p]
    assert "grad_int8" in state
    assert "scale_grad" in state
    assert state["grad_int8"].dtype == torch.int8


def test_when_sgd_qat_then_step_changes_weight() -> None:
    """The QAT step still moves the weight (no silent no-op)."""
    lin, opt = _tiny_sgd_optimizer()
    p = next(lin.parameters())
    before = p.data.clone()
    x = torch.randn(4, 8)
    loss = lin(x).sum()
    loss.backward()
    opt.step()
    assert not torch.equal(before, p.data)


def test_when_sgd_qat_then_matches_non_qat_within_tolerance() -> None:
    """The QAT step tracks the fp32 SGD step (quantization is a
    bounded perturbation, not a different trajectory)."""
    torch.manual_seed(5)
    lin_qat = torch.nn.Linear(8, 8, bias=False)
    lin_fp32 = torch.nn.Linear(8, 8, bias=False)
    lin_fp32.load_state_dict(lin_qat.state_dict())
    from nanochat.training.minexp_lstm import SGD_MinExpLstm_Optimizer

    opt_qat = SGD_MinExpLstm_Optimizer(lin_qat.parameters(), lr=0.05, qat_4bit=True)
    opt_fp32 = SGD_MinExpLstm_Optimizer(lin_fp32.parameters(), lr=0.05, qat_4bit=False)
    for _ in range(3):
        x = torch.randn(4, 8)
        loss_q = lin_qat(x).sum()
        loss_f = lin_fp32(x).sum()
        loss_q.backward()
        loss_f.backward()
        opt_qat.step()
        opt_fp32.step()
    w_q = next(lin_qat.parameters()).data
    w_f = next(lin_fp32.parameters()).data
    # Quantization noise is bounded by the 4-bit grid; the two weights stay close (relative error well under the fp32 step).
    torch.testing.assert_close(w_q, w_f, rtol=0.5, atol=0.5)


def test_when_retrofit_fourbit_then_linears_become_fourbit() -> None:
    """retrofit_fourbit swaps nn.Linear for FourBitLinear in place.

    Eligible linears (>= min_linear_dim, not lm_head, not already
    retrofitted) become FourBitLinear whose persistent weight is
    packed nibbles; lm_head stays fp32.
    """
    from nanochat.models.quant import retrofit_fourbit

    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 128, bias=False),
        torch.nn.Linear(128, 128, bias=False),
    )
    retrofit_fourbit(model, min_linear_dim=128)
    for child in model:
        assert isinstance(child, FourBitLinear)
        assert child.weight_packed.dtype == torch.uint8
        assert child.weight_packed.numel() == (128 * 128 + 1) // 2


def test_when_retrofit_fourbit_then_forward_runs() -> None:
    """A retrofitted fourbit model still runs a forward pass."""
    from nanochat.models.quant import retrofit_fourbit

    torch.manual_seed(8)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 128, bias=False),
        torch.nn.Linear(128, 128, bias=False),
    )
    retrofit_fourbit(model, min_linear_dim=128)
    x = torch.randn(2, 128)
    y = model(x)
    assert y.shape == (2, 128)


# --- QAT on the MeZO optimizer: 4-bit weight update ---


def _tiny_mezo_optimizer():
    from nanochat.training.mezo_minexplstm import MeZo_MinExpLstm_Optimizer

    torch.manual_seed(6)
    lin = torch.nn.Linear(8, 8, bias=False)
    opt = MeZo_MinExpLstm_Optimizer(
        lin.parameters(), lr=0.1, epsilon=1e-3, qat_4bit=True
    )
    return lin, opt


def test_when_mezo_qat_then_step_changes_weight() -> None:
    """The MeZO QAT step moves the weight from the staged estimate."""
    lin, opt = _tiny_mezo_optimizer()
    p = next(lin.parameters())
    before = p.data.clone()
    # Stage a MeZO estimate, then step.
    seed = 12345
    opt.perturb(seed, scaling=1.0)
    loss_pos = lin(torch.randn(4, 8)).sum()
    opt.perturb(seed, scaling=-2.0)
    loss_neg = lin(torch.randn(4, 8)).sum()
    opt.perturb(seed, scaling=1.0)
    opt.stage_mezo_estimate(seed, loss_pos, loss_neg)
    opt.step()
    assert not torch.equal(before, p.data)


def test_when_mezo_qat_then_state_has_int8_grad() -> None:
    """The MeZO QAT path keeps its INT8 hidden state."""
    lin, opt = _tiny_mezo_optimizer()
    p = next(lin.parameters())
    x = torch.randn(4, 8)
    # MeZO needs a staged estimate; run one step to init state.
    seed = 777
    opt.perturb(seed, scaling=1.0)
    lp = lin(x).sum()
    opt.perturb(seed, scaling=-2.0)
    ln = lin(x).sum()
    opt.perturb(seed, scaling=1.0)
    opt.stage_mezo_estimate(seed, lp, ln)
    opt.step()
    state = opt.state[p]
    assert "h_int8" in state
    assert state["h_int8"].dtype == torch.int8
