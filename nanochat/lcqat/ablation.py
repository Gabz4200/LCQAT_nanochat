"""
Quantization-range ablation primitives: symmetric falloff windows and learnable
grid positions.

`falloff`, and the `gaussian_falloff` / `cosine_falloff` / `quartic_falloff`
kernels behind it, weight a 1-D axis by a bump that is exactly 1.0 at a center
and decays to roughly 0 at `radius` on both sides of it (`center + radius` and
`center - radius`). They are elementwise, bounded in [0, 1] and differentiable
in the axis, the center and the radius, so any of the three can be learned.

`StrictLearnableGrid` exposes a fixed number of grid positions whose endpoints
are learned. Because both endpoints are trainable, every derived quantity is
reparameterized so that no optimizer step can collapse the grid (inverted,
overlapping or duplicated positions); see `StrictLearnableGrid` for the exact
guarantees.
"""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_float(name: str, value: float) -> float:
    """Validate that `value` is a finite real number. Returns it as a float."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Return `x` such that `F.softplus(x) == value`, for `value > 0`.

    Uses `log(-expm1(-value))` rather than `log(expm1(value))` so that large
    spans (above ~88 in FP32, where `expm1` overflows to inf) stay finite.
    """
    return value + torch.log(-torch.expm1(-value))


class StrictLearnableGrid(nn.Module):
    """A strictly increasing grid of `n_points` learnable positions.

    `low` and the grid span are optimized *indirectly*: the forward pass
    reparameterizes both so that no optimizer step can collapse the grid.

    * `span = intervals * min_spacing + softplus(raw_span)` (inverse-softplus
      init, the same trick as `nanochat.lcqat.codebook`): the domain is always
      strictly longer than the minimum span the spacing constraint permits, so
      the slack shared out over the intervals can never go negative. A directly
      learned `high - low` could shrink below `intervals * min_spacing` and emit
      inverted or overcrowded positions.
    * `gap_i = min_spacing + slack * weight_i`, where `weight` is a floored
      softmax over `raw_weights`: every weight stays positive, so neighbouring
      positions remain strictly ordered even when `min_spacing` is 0, where a
      saturated softmax would collapse some gaps to exactly 0.
    * `low` stays a free translation; shifting the whole grid cannot collapse
      it, so it needs no reparameterization.

    The parameters are named `low`, `raw_span` and `raw_weights`; the two raw
    names are the shape-stable contract for callers that want to give this
    module its own optimizer group (see `nanochat.lcqat.optimizer`).

    Args:
        n_points: number of grid positions, at least 2.
        low_init: initial position of the lowest grid point.
        high_init: initial position of the highest grid point.
        min_spacing: minimum distance between neighbouring positions.
        dtype: dtype of the parameters and of the returned positions.
        weight_floor: smallest share of the slack handed to each interval, a
            convex mixture of `softmax(raw_weights)` with the uniform
            distribution, in `[0, 1 / (n_points - 1)]`. `0.0` recovers the plain
            softmax, which is only collapse-free when `min_spacing > 0`. With
            `min_spacing == 0` the floor also has to stay above the axis
            resolution of the grid (`torch.finfo(dtype).eps` scaled by the grid
            magnitude), otherwise the guarded gaps round back to duplicates.

    Raises:
        ValueError: on out-of-range arguments, on undecodable initial spans, or
            when the requested initial grid is not strictly longer than
            `(n_points - 1) * min_spacing`.
    """

    def __init__(
        self,
        n_points: int,
        low_init: float,
        high_init: float,
        min_spacing: float,
        dtype: torch.dtype = torch.float32,
        weight_floor: float = 1e-4,
    ) -> None:
        super().__init__()

        if isinstance(n_points, bool) or not isinstance(n_points, int):
            raise ValueError(f"n_points must be an int, got {n_points!r}")
        if n_points < 2:
            raise ValueError(f"n_points must be at least 2, got {n_points}")

        self.n_points = n_points
        self.intervals = n_points - 1

        self.min_spacing = _finite_float("min_spacing", min_spacing)
        if self.min_spacing < 0:
            raise ValueError(f"min_spacing must be non-negative, got {min_spacing}")
        self.min_span = self.intervals * self.min_spacing

        self.weight_floor = _finite_float("weight_floor", weight_floor)
        max_floor = 1.0 / self.intervals
        if not 0.0 <= self.weight_floor <= max_floor:
            raise ValueError(
                f"weight_floor must be in [0, {max_floor}], got {weight_floor}"
            )

        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be floating point, got {dtype}")

        low_init = _finite_float("low_init", low_init)
        high_init = _finite_float("high_init", high_init)
        if high_init <= low_init:
            raise ValueError(
                f"high_init must be greater than low_init, got "
                f"{high_init} <= {low_init}"
            )

        # Slack must be strictly positive: a zero (or negative) slack has no
        # inverse softplus, it would put raw_span at -inf and freeze the grid at
        # exactly min_spacing, which is the collapsed configuration this class
        # exists to avoid.
        slack = high_init - low_init - self.min_span
        if slack <= 0.0:
            raise ValueError(
                f"high_init - low_init must exceed "
                f"(n_points - 1) * min_spacing = {self.min_span}, got "
                f"{high_init - low_init}"
            )

        raw_span = _inverse_softplus(torch.tensor(slack, dtype=dtype))
        if not torch.isfinite(raw_span):
            raise ValueError(
                f"init slack {slack} is not representable in dtype {dtype}; use a "
                f"larger high_init - low_init or a smaller min_spacing"
            )

        self.low = nn.Parameter(torch.tensor(low_init, dtype=dtype))
        self.raw_span = nn.Parameter(raw_span)
        self.raw_weights = nn.Parameter(torch.zeros(self.intervals, dtype=dtype))

    @property
    def domain_length(self) -> torch.Tensor:
        """Learned span `high - low`, always strictly greater than `min_span`."""
        return self.min_span + F.softplus(self.raw_span)

    @property
    def high(self) -> torch.Tensor:
        """Learned position of the highest grid point."""
        return self.low + self.domain_length

    def forward(self) -> torch.Tensor:
        """Return the grid as a strictly increasing `(n_points,)` tensor."""
        slack = F.softplus(self.raw_span)
        gaps = self.min_spacing + slack * self._slack_weights()
        positions = torch.cumsum(torch.cat([gaps.new_zeros(1), gaps]), dim=0)
        return self.low + positions

    def _slack_weights(self) -> torch.Tensor:
        """Share the slack over the intervals: a floored softmax of the logits.

        `floor + (1 - intervals * floor) * softmax` is a convex combination that
        keeps every weight at least `floor`, so every gap is at least
        `min_spacing + floor * slack > 0` and the grid cannot collapse.
        """
        weights = F.softmax(self.raw_weights, dim=0)
        floor = self.weight_floor
        if floor == 0.0:
            return weights
        return floor + (1.0 - self.intervals * floor) * weights


# A falloff argument: a Python number, or a scalar/broadcastable tensor (which
# keeps the center and the radius learnable).
FalloffScalar = float | torch.Tensor

# Signature shared by the kernels and the registry: (x, center, radius) -> Tensor.
FalloffFn = Callable[[torch.Tensor, FalloffScalar, FalloffScalar], torch.Tensor]


def _scalar_arg(name: str, value: FalloffScalar) -> FalloffScalar:
    """Validate a falloff scalar: a finite number, or a floating-point tensor."""
    if isinstance(value, torch.Tensor):
        if not value.dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point, got dtype {value.dtype}")
        return value
    return _finite_float(name, value)


def _falloff_offset(
    x: torch.Tensor, center: FalloffScalar, radius: FalloffScalar
) -> torch.Tensor:
    """Return `(x - center) / radius`: the signed distance measured in radii.

    Shared contract of every falloff kernel: `x` is a 1-D floating-point axis,
    `center` and `radius` are finite numbers or broadcastable tensors, and the
    radius is strictly positive. Only a numeric radius is checked here; a tensor
    radius is the caller's job to keep positive (the same contract as the learned
    spans elsewhere in LC-QAT), so gradients can flow through it unchecked.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {tuple(x.shape)}")
    if not x.dtype.is_floating_point:
        raise ValueError(f"x must be floating point, got dtype {x.dtype}")
    center = _scalar_arg("center", center)
    radius = _scalar_arg("radius", radius)
    if isinstance(radius, float) and radius <= 0.0:
        raise ValueError(f"radius must be strictly positive, got {radius}")
    return (x - center) / radius


def gaussian_falloff(
    x: torch.Tensor,
    center: FalloffScalar,
    radius: FalloffScalar,
    *,
    edge: float = 1e-2,
) -> torch.Tensor:
    r"""Gaussian bump calibrated so `radius` is the half-width of the window.

    `exp(-(d / sigma)^2 / 2)` with `d = x - center` and
    `sigma = radius / sqrt(2 * ln(1 / edge))`, so the value is exactly 1.0 at
    `center`, `edge` at `center +- radius`, and decays asymptotically beyond it.
    A gaussian never reaches exactly 0, so use `cosine_falloff` or
    `quartic_falloff` when a compact window is needed.

    Args:
        x: 1-D floating-point axis.
        center: falloff center, a Python number or a broadcastable tensor.
        radius: distance from `center` at which the value has decayed to `edge`.
        edge: value at `|x - center| == radius`, strictly inside (0, 1). Smaller
            values give a narrower bump for the same `radius`;
            `edge=exp(-0.5)` recovers the textbook convention where `radius`
            is the standard deviation.

    Returns:
        Tensor shaped like `x` (broadcast with `center` / `radius`), elementwise
        in (0, 1], exactly 1.0 where `x == center`.
    """
    edge = _finite_float("edge", edge)
    if not 0.0 < edge < 1.0:
        raise ValueError(f"edge must be in (0, 1), got {edge}")
    offset = _falloff_offset(x, center, radius)
    scaled = math.sqrt(-2.0 * math.log(edge)) * offset
    return torch.exp(-0.5 * scaled * scaled)


def cosine_falloff(
    x: torch.Tensor, center: FalloffScalar, radius: FalloffScalar
) -> torch.Tensor:
    """Raised-cosine (Hann) bump: `0.5 * (1 + cos(pi * (x - center) / radius))`.

    Exactly 1.0 at `center`, exactly 0.0 at and beyond `|x - center| == radius`.
    The offset is clamped into the window, which loses nothing: the slope already
    vanishes at `+-radius`, so the flat tail carries the same (zero) gradient the
    unclamped formula would give, without evaluating `cos` on an oscillating tail.

    Args:
        x: 1-D floating-point axis.
        center: falloff center, a Python number or a broadcastable tensor.
        radius: half-width of the window.

    Returns:
        Tensor shaped like `x` (broadcast with `center` / `radius`), elementwise
        in [0, 1], exactly 1.0 at `center` and exactly 0.0 outside the window.
    """
    offset = _falloff_offset(x, center, radius)
    return 0.5 * (1.0 + torch.cos(math.pi * offset.clamp(-1.0, 1.0)))


def quartic_falloff(
    x: torch.Tensor, center: FalloffScalar, radius: FalloffScalar
) -> torch.Tensor:
    """Quartic (Welch) bump: `(1 - u^2)^2` inside `|u| <= 1`, else 0.

    Exactly 1.0 at `center`, exactly 0.0 at and beyond `|x - center| == radius`.
    Compared with `cosine_falloff` it keeps more of its weight near the center and
    joins the axis more flatly, so the same `radius` gives a more peaked window.

    Args:
        x: 1-D floating-point axis.
        center: falloff center, a Python number or a broadcastable tensor.
        radius: half-width of the window.

    Returns:
        Tensor shaped like `x` (broadcast with `center` / `radius`), elementwise
        in [0, 1], exactly 1.0 at `center` and exactly 0.0 outside the window.
    """
    offset = _falloff_offset(x, center, radius)
    residual = (1.0 - offset * offset).clamp_min(0.0)
    return residual * residual


# Falloff registry: name -> symmetric elementwise window over a 1-D axis.
FALLOFFS: dict[str, FalloffFn] = {
    "gaussian": gaussian_falloff,
    "cosine": cosine_falloff,
    "quartic": quartic_falloff,
}


def falloff(
    x: torch.Tensor,
    center: FalloffScalar,
    radius: FalloffScalar,
    kind: str = "gaussian",
) -> torch.Tensor:
    """Evaluate the symmetric falloff window `kind` over the 1-D axis `x`.

    Args:
        x: 1-D floating-point axis.
        center: falloff center, a Python number or a broadcastable tensor.
        radius: half-width of the window, as interpreted by the selected kind.
        kind: a key of `FALLOFFS`: "gaussian", "cosine" or "quartic".

    Returns:
        Tensor shaped like `x` (broadcast with `center` / `radius`), elementwise
        in [0, 1], exactly 1.0 at `center`.

    Raises:
        ValueError: if `kind` is not a registered falloff.
    """
    if kind not in FALLOFFS:
        raise ValueError(
            f"Unknown falloff kind: {kind!r}. Valid kinds: {sorted(FALLOFFS)}"
        )
    return FALLOFFS[kind](x, center, radius)
