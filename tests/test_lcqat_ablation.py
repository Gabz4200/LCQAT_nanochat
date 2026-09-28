"""
Tests for the ablation primitives: symmetric falloff windows and the learnable
grid.

python -m pytest tests/test_lcqat_ablation.py -v
"""

from collections.abc import Callable

import pytest
import torch

from nanochat.lcqat.ablation import (
    FALLOFFS,
    StrictLearnableGrid,
    cosine_falloff,
    falloff,
    gaussian_falloff,
    quartic_falloff,
)

FALLOFF_KINDS = sorted(FALLOFFS)


def make_grid(
    n_points: int = 9,
    min_spacing: float = 0.05,
    low_init: float = -1.0,
    high_init: float = 1.0,
    **kwargs,
) -> StrictLearnableGrid:
    return StrictLearnableGrid(
        n_points=n_points,
        low_init=low_init,
        high_init=high_init,
        min_spacing=min_spacing,
        **kwargs,
    )


def assert_strict_grid(grid: StrictLearnableGrid) -> torch.Tensor:
    """Assert the grid contract holds; returns the detached positions."""
    positions = grid().detach()
    gaps = positions[1:] - positions[:-1]
    # Positions are a cumsum, so differencing neighbours reintroduces a few ULPs
    # of the grid magnitude; the spacing guarantee is exact in real arithmetic.
    tol = 4 * torch.finfo(positions.dtype).eps * max(1.0, positions.abs().max().item())
    assert torch.isfinite(positions).all()
    assert (gaps > 0).all(), f"grid collapsed: {positions.tolist()}"
    assert (gaps >= grid.min_spacing - tol).all()
    # In the saturated limit softplus(raw_span) underflows to 0, so the span only
    # equals min_span (it is strictly greater right after init / while training
    # with usable gradients).
    assert grid.domain_length.detach().item() >= grid.min_span
    return positions


def test_when_constructed_then_init_grid_is_uniform_between_the_endpoints() -> None:
    for n_points in (2, 9, 255):
        grid = make_grid(n_points=n_points, min_spacing=0.0)
        positions = assert_strict_grid(grid)
        assert positions.shape == (n_points,)
        assert positions.dtype == torch.float32
        assert positions[0].item() == pytest.approx(-1.0, abs=1e-6)
        assert positions[-1].item() == pytest.approx(1.0, abs=1e-6)
        assert grid.domain_length.item() > grid.min_span
        gaps = positions[1:] - positions[:-1]
        assert torch.allclose(
            gaps,
            positions.new_full((n_points - 1,), 2.0 / (n_points - 1)),
            rtol=1e-5,
            atol=1e-6,
        )


def test_when_forward_then_high_matches_the_last_position() -> None:
    grid = make_grid(n_points=17, min_spacing=0.01)
    positions = grid()
    assert positions[0].item() == grid.low.item()
    assert positions[-1].item() == pytest.approx(grid.high.item(), rel=1e-5, abs=1e-6)
    assert grid.high.item() - grid.low.item() == pytest.approx(
        grid.domain_length.item(), rel=1e-6
    )


def test_when_optimizer_drives_the_span_down_then_grid_never_collapses() -> None:
    # Regression: with a freely learned `high - low`, the span could shrink below
    # (n_points - 1) * min_spacing, so `available` went negative and the gaps
    # became negative/overlapping (an inverted grid).
    grid = make_grid(min_spacing=0.05)
    opt = torch.optim.Adam(grid.parameters(), lr=0.5)
    span_seen = float("inf")
    for _ in range(300):
        opt.zero_grad()
        positions = grid()
        (positions[-1] - positions[0]).backward()  # pull the span down hard
        opt.step()
        span_seen = min(span_seen, grid.domain_length.item())
        assert_strict_grid(grid)
    # The optimizer stalls on Adam's eps floor just above the constraint, but the
    # span never crosses it (the old formula went to 0 and then negative).
    assert span_seen >= grid.min_span
    assert grid.min_span < grid.domain_length.item() < grid.min_span * 1.01


def test_when_optimizer_translates_the_grid_then_spacing_is_preserved() -> None:
    grid = make_grid()
    opt = torch.optim.Adam(grid.parameters(), lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        grid()[0].neg().backward()  # push `low` up
        opt.step()
        positions = assert_strict_grid(grid)
    assert positions[0].item() > 0.0


def test_when_min_spacing_is_zero_then_floored_softmax_keeps_positions_distinct() -> (
    None
):
    # With min_spacing == 0 the only thing keeping the grid strict is the weight
    # floor: a saturated softmax would zero every gap but the largest.
    grid = make_grid(min_spacing=0.0, weight_floor=1e-4)
    with torch.no_grad():
        grid.raw_weights[0] = 1000.0
    positions = assert_strict_grid(grid)
    gaps = positions[1:] - positions[:-1]
    assert gaps.min().item() > 0.0


def test_when_min_spacing_is_positive_then_plain_softmax_is_still_strict() -> None:
    grid = make_grid(min_spacing=0.05, weight_floor=0.0)
    with torch.no_grad():
        grid.raw_weights[0] = 1000.0
    assert_strict_grid(grid)


def test_when_span_parameter_is_saturated_then_grid_stays_strict() -> None:
    for raw_span in (-1e6, 1e6):
        grid = make_grid(min_spacing=0.05)
        with torch.no_grad():
            grid.raw_span.fill_(raw_span)
        positions = assert_strict_grid(grid)
        if raw_span < 0:
            span = positions[-1].item() - positions[0].item()
            assert span == pytest.approx(grid.min_span, rel=1e-5)


def test_when_backward_then_all_parameters_receive_finite_gradients() -> None:
    grid = make_grid()
    grid().sum().backward()
    for name in ("low", "raw_span", "raw_weights"):
        grad = getattr(grid, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_when_state_dict_then_parameter_names_are_the_stable_contract() -> None:
    grid = make_grid()
    assert sorted(grid.state_dict()) == ["low", "raw_span", "raw_weights"]
    clone = make_grid()
    clone.load_state_dict(grid.state_dict())
    assert torch.equal(clone(), grid())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(n_points=1), "at least 2"),
        (dict(n_points=9.0), "must be an int"),
        (dict(min_spacing=-0.1), "non-negative"),
        (dict(min_spacing=float("inf")), "finite number"),
        (dict(low_init=float("nan")), "finite number"),
        (dict(low_init=1.0, high_init=1.0), "greater than"),
        (dict(high_init=-2.0), "greater than"),
        (dict(dtype=torch.int64), "floating point"),
        # span 2.0 does not exceed (n_points - 1) * 0.25 = 2.25
        (dict(min_spacing=0.25), "must exceed"),
        (dict(weight_floor=-1e-5), "weight_floor"),
        (dict(weight_floor=0.2), "weight_floor"),
        # softplus inverse overflows float16 for a 2e6 span
        (dict(dtype=torch.float16, low_init=-1e6, high_init=1e6), "not representable"),
    ],
)
def test_when_arguments_are_invalid_then_raises_value_error(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        make_grid(**kwargs)


def test_when_dtype_is_float64_then_parameters_and_positions_match() -> None:
    grid = make_grid(dtype=torch.float64)
    positions = assert_strict_grid(grid)
    assert positions.dtype == torch.float64
    for name in ("low", "raw_span", "raw_weights"):
        assert getattr(grid, name).dtype == torch.float64


def test_when_dtype_is_bfloat16_then_grid_stays_strict() -> None:
    grid = make_grid(dtype=torch.bfloat16)
    positions = assert_strict_grid(grid)
    assert positions.dtype == torch.bfloat16
    assert grid.raw_weights.dtype == torch.bfloat16


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_at_center_then_falloff_is_exactly_one(kind: str) -> None:
    for center, radius in ((0.0, 0.5), (-1.25, 2.0), (3.0, 0.5)):
        at_center = torch.tensor([center], dtype=torch.float64)
        assert falloff(at_center, center, radius, kind).item() == 1.0


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_at_the_window_edge_then_falloff_has_decayed(kind: str) -> None:
    # A gaussian only approaches 0 (it is exactly `edge` at the window edge); the
    # compact windows are 0 there.
    edge_value = 1e-2 if kind == "gaussian" else 0.0
    for center, radius in ((0.0, 0.5), (-3.25, 2.0), (1.5, 0.25)):
        left = falloff(torch.tensor([center - radius]), center, radius, kind)
        right = falloff(torch.tensor([center + radius]), center, radius, kind)
        assert left.item() == pytest.approx(edge_value, rel=1e-5, abs=1e-6)
        assert right.item() == pytest.approx(edge_value, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("kind", ["cosine", "quartic"])
def test_when_at_or_beyond_the_edge_then_compact_falloffs_are_exactly_zero(
    kind: str,
) -> None:
    for offset in (1.0, 1.0 + 1e-3, 2.0, 1e3):
        for sign in (-1.0, 1.0):
            assert falloff(torch.tensor([sign * offset]), 0.0, 1.0, kind).item() == 0.0


def test_when_far_from_the_center_then_gaussian_underflows_without_nan() -> None:
    values = gaussian_falloff(torch.tensor([-1e3, 1e3]), 0.0, 1.0)
    assert torch.isfinite(values).all()
    assert torch.equal(values, torch.zeros_like(values))


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_sweeping_the_axis_then_falloff_is_symmetric_monotone_and_bounded(
    kind: str,
) -> None:
    offsets = torch.linspace(-3.0, 3.0, 601)
    values = falloff(offsets, 0.0, 1.0, kind)
    assert values.shape == offsets.shape
    assert (values >= 0.0).all() and (values <= 1.0).all()
    assert torch.allclose(values, values.flip(0))
    middle = values.numel() // 2
    assert values[middle].item() == pytest.approx(1.0, rel=1e-6)
    assert (values[middle:].diff() <= 0.0).all()


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_radius_grows_then_the_window_widens(kind: str) -> None:
    x = torch.tensor([0.75])
    assert falloff(x, 0.0, 1.0, kind).item() < falloff(x, 0.0, 2.0, kind).item()


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_backward_through_the_axis_then_all_three_inputs_get_gradients(
    kind: str,
) -> None:
    x = torch.linspace(-1.0, 1.0, 11, requires_grad=True)
    center = torch.tensor(0.25, requires_grad=True)
    radius = torch.tensor(0.5, requires_grad=True)
    # An asymmetric loss, otherwise the center gradient cancels over the grid.
    weights = torch.linspace(0.0, 1.0, 11)
    (falloff(x, center, radius, kind) * weights).sum().backward()
    grads = (("x", x.grad), ("center", center.grad), ("radius", radius.grad))
    for name, grad in grads:
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_center_and_radius_are_tensors_then_gradients_flow_through_them(
    kind: str,
) -> None:
    x = torch.linspace(-1.0, 1.0, 7)
    center = torch.full_like(x, 0.2, requires_grad=True)
    radius = torch.full_like(x, 0.4, requires_grad=True)
    values = falloff(x, center, radius, kind)
    assert values.shape == x.shape
    assert values.dtype == x.dtype
    values.sum().backward()
    for grad in (center.grad, radius.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["cosine", "quartic"])
def test_when_outside_the_window_then_value_and_gradient_are_exactly_zero(
    kind: str,
) -> None:
    x = torch.tensor([-5.0, 5.0], requires_grad=True)
    values = falloff(x, 0.0, 1.0, kind)
    values.sum().backward()
    assert torch.equal(values, torch.zeros_like(values))
    assert x.grad is not None
    assert torch.equal(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize("kind", FALLOFF_KINDS)
def test_when_axis_dtype_changes_then_values_follow_it_and_stay_bounded(
    dtype: torch.dtype, kind: str
) -> None:
    x = torch.linspace(-2.0, 2.0, 9, dtype=dtype)
    values = falloff(x, 0.0, 0.5, kind)
    assert values.dtype == dtype
    assert torch.isfinite(values).all()
    assert (values >= 0.0).all() and (values <= 1.0).all()


def test_when_gaussian_edge_changes_then_it_sets_the_value_at_radius() -> None:
    for edge in (1e-1, 1e-2, 1e-4):
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        values = gaussian_falloff(x, 0.0, 0.5, edge=edge)
        assert values[1].item() == 1.0
        assert values[0].item() == pytest.approx(edge, rel=1e-6)
        assert values[2].item() == pytest.approx(edge, rel=1e-6)


@pytest.mark.parametrize(
    ("kind", "kernel"),
    (
        ("gaussian", gaussian_falloff),
        ("cosine", cosine_falloff),
        ("quartic", quartic_falloff),
    ),
)
def test_when_a_kind_is_requested_then_the_dispatcher_calls_its_kernel(
    kind: str, kernel: Callable[[torch.Tensor, float, float], torch.Tensor]
) -> None:
    x = torch.linspace(-2.0, 2.0, 13)
    assert torch.equal(falloff(x, 0.3, 0.7, kind), kernel(x, 0.3, 0.7))


@pytest.mark.parametrize(
    ("call", "match"),
    (
        (lambda: falloff(torch.zeros(2, 2), 0.0, 1.0), "must be 1-D"),
        (lambda: falloff(torch.arange(4), 0.0, 1.0), "floating point"),
        (lambda: falloff([0.0, 1.0], 0.0, 1.0), "must be a torch.Tensor"),
        (lambda: falloff(torch.zeros(4), 0.0, 0.0), "strictly positive"),
        (lambda: falloff(torch.zeros(4), 0.0, -1.0), "strictly positive"),
        (lambda: falloff(torch.zeros(4), 0.0, float("nan")), "finite number"),
        (lambda: falloff(torch.zeros(4), float("inf"), 1.0), "finite number"),
        (
            lambda: falloff(torch.zeros(4), 0.0, torch.ones(4, dtype=torch.int64)),
            "floating point",
        ),
        (lambda: falloff(torch.zeros(4), 0.0, 1.0, "triangle"), "Unknown falloff kind"),
        (lambda: gaussian_falloff(torch.zeros(4), 0.0, 1.0, edge=0.0), "edge"),
        (lambda: gaussian_falloff(torch.zeros(4), 0.0, 1.0, edge=1.0), "edge"),
    ),
)
def test_when_falloff_arguments_are_invalid_then_raises_value_error(
    call: Callable[[], torch.Tensor], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        call()
