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
from typing import Literal

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


# Learnable Luts


def _validate_float_tensor(name: str, value: torch.Tensor) -> None:
    """Validate that `value` is a floating-point tensor."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point, got dtype {value.dtype}")


def _prepare_weight_floor(n_points: int, weight_floor: float) -> float:
    """Validate and clamp weight_floor to the maximum legal value.

    This avoids the default floor becoming invalid for very large grids where
    1 / (n_points - 1) is smaller than the requested default.
    """
    weight_floor = _finite_float("weight_floor", weight_floor)

    if isinstance(n_points, int) and not isinstance(n_points, bool) and n_points >= 2:
        max_floor = 1.0 / (n_points - 1)
        if weight_floor > max_floor:
            weight_floor = max_floor

    return weight_floor


def _grid_temperature(grid: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Return a stable soft-assignment temperature/radius from a grid.

    The returned value is:

        temperature = mean_gap / sharpness

    This is used both for softmax-style soft assignment and as the radius for
    falloff kernels. Larger `sharpness` gives narrower / more local kernels.
    """
    grid32 = grid.detach().to(torch.float32)

    if grid32.numel() < 2:
        return torch.tensor(
            torch.finfo(torch.float32).eps,
            dtype=torch.float32,
            device=grid32.device,
        )

    gaps = grid32[1:] - grid32[:-1]
    temp = gaps.mean() / sharpness

    # Scale-aware lower bound. This avoids zero temperature/radius for tiny grids.
    scale = torch.clamp_min(grid32.abs().amax(), torch.finfo(torch.float32).tiny)
    eps = torch.finfo(torch.float32).eps * scale

    return torch.clamp_min(temp, eps)


def _soft_codebook_weights(
    x: torch.Tensor,
    centers: torch.Tensor,
    sharpness: float,
) -> torch.Tensor:
    """Differentiable soft-assignment weights over 1-D centers.

    Args:
        x: arbitrary-shaped floating-point tensor.
        centers: 1-D strictly increasing tensor of codebook/knot positions.
        sharpness: positive scalar. Larger means sharper / more local.

    Returns:
        Weights shaped `x.shape + (centers.shape[0],)`, nonnegative and summing
        to 1 over the last dimension.

    Notes:
        This uses a stabilized relative squared-distance kernel:

            d_i = |x - c_i| / temperature
            m   = min_i d_i
            w_i ∝ exp(-(max(0, d_i - m)^2))

        Subtracting the minimum distance makes the softmax stable and avoids
        arbitrary absolute distance scaling. Clamping the input into the center
        range makes out-of-range behavior stable and gives endpoint saturation.
    """
    x32 = x.to(torch.float32)
    c32 = centers.to(torch.float32)

    if c32.numel() == 0:
        return torch.empty(
            x32.shape + (0,),
            dtype=torch.float32,
            device=x32.device,
        )

    if c32.numel() == 1:
        return torch.ones(
            x32.shape + (1,),
            dtype=torch.float32,
            device=x32.device,
        )

    # Clamp inputs into the learned domain before computing distances.
    # This avoids inf-driven NaNs and gives sane endpoint behavior.
    x_clamped = torch.minimum(torch.maximum(x32, c32[0]), c32[-1])

    temp = _grid_temperature(c32, sharpness)

    # Shape: x.shape + (K,)
    dist = torch.abs(x_clamped.unsqueeze(-1) - c32) / temp

    # Relative distances to the nearest center.
    d_min = dist.amin(dim=-1, keepdim=True)
    rel = dist - d_min

    # Keep logits in a sane range. sqrt(50) gives logits >= -50,
    # which is already effectively zero weight but avoids overflow/underflow.
    rel = torch.clamp(rel, min=0.0, max=math.sqrt(50.0))

    logits = -rel * rel
    return F.softmax(logits, dim=-1)


def _falloff_weights(
    x: torch.Tensor,
    centers: torch.Tensor,
    sharpness: float,
    kind: str,
) -> torch.Tensor:
    """Evaluate unnormalized falloff weights for every input against every center.

    Args:
        x: arbitrary-shaped floating-point tensor.
        centers: 1-D tensor of center positions.
        sharpness: positive scalar. The falloff radius is derived as
            `radius = mean_gap / sharpness`.
        kind: falloff kind from `FALLOFFS`.

    Returns:
        Unnormalized weights shaped `x.shape + (centers.shape[0],)`.
    """
    if kind not in FALLOFFS:
        raise ValueError(
            f"Unknown falloff kind: {kind!r}. Valid kinds: {sorted(FALLOFFS)}"
        )

    x32 = x.to(torch.float32)
    c32 = centers.to(torch.float32)

    k = c32.numel()
    out_shape = x32.shape + (k,)

    if k == 0:
        return torch.empty(out_shape, dtype=torch.float32, device=x32.device)

    x_flat = x32.reshape(-1)
    n = x_flat.numel()

    if n == 0:
        return torch.empty(out_shape, dtype=torch.float32, device=x32.device)

    # Reuse `sharpness` to derive the falloff radius.
    radius = _grid_temperature(c32, sharpness)

    # The existing `falloff()` implementation expects a 1-D axis. Flatten the
    # pairwise problem into a 1-D problem, then reshape back.
    x_expanded = x_flat.unsqueeze(-1).expand(n, k).reshape(-1)
    c_expanded = c32.unsqueeze(0).expand(n, k).reshape(-1)

    w_flat = falloff(
        x_expanded,
        c_expanded,
        radius,
        kind=kind,
    )
    w = w_flat.reshape(n, k)

    return w.reshape(out_shape)


def _piecewise_linear_lookup(
    x: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
) -> torch.Tensor:
    """Differentiable piecewise-linear lookup.

    The learned function is defined by points `(xs[i], ys[i])`, where `xs` is
    strictly increasing. Inputs outside `[xs[0], xs[-1]]` clamp to the endpoint
    values.

    This is differentiable with respect to:
      - input `x` inside intervals,
      - knot positions `xs`,
      - knot values `ys`.

    The index selection itself is discrete, as in any piecewise-linear function.
    """
    x_flat = x.reshape(-1).to(torch.float32)
    xs32 = xs.to(torch.float32)
    ys32 = ys.to(torch.float32)

    if xs32.numel() == 0:
        return torch.empty(x.shape, dtype=ys.dtype, device=ys.device)

    if xs32.numel() == 1:
        return torch.full_like(x, ys32[0], dtype=ys32.dtype).to(ys.dtype)

    # First index with xs[idx] >= x.
    idx = torch.searchsorted(xs32, x_flat)

    # Use interval [idx - 1, idx]. Clamp handles below-range and above-range.
    idx = idx.clamp(1, xs32.numel() - 1)

    x0 = xs32[idx - 1]
    x1 = xs32[idx]
    y0 = ys32[idx - 1]
    y1 = ys32[idx]

    denom = x1 - x0

    # StrictLearnableGrid should make denom > 0. This guard avoids division by
    # zero in pathological dtype/rounding cases.
    scale = torch.clamp_min(
        torch.maximum(x0.abs(), x1.abs()),
        torch.finfo(xs32.dtype).tiny,
    )
    eps = torch.finfo(xs32.dtype).eps * scale
    denom = torch.clamp_min(denom, eps)

    t = ((x_flat - x0) / denom).clamp(0.0, 1.0)
    out = y0 + t * (y1 - y0)

    return out.reshape(x.shape).to(ys.dtype)


class LearnableLinearLut(nn.Module):
    """Differentiable soft codebook lookup.

    This maps an input tensor to a convex combination of learned codebook
    entries. The codebook entries are kept strictly increasing by
    `StrictLearnableGrid`.

    Modes:
        "softmax":
            Soft nearest-neighbor lookup using a stabilized distance softmax.

        "falloff":
            Fallback-based soft lookup using normalized falloff weights. If
            compact-support falloffs produce zero mass, the module falls back
            to the softmax weights so the output remains well-defined.

    This module is shape-agnostic and returns a tensor with the same shape as
    the input.

    Args:
        n_points: number of codebook entries.
        low_init: initial lowest codebook value.
        high_init: initial highest codebook value.
        min_spacing: minimum spacing between codebook entries.
        dtype: dtype of the learned codebook and returned output.
        weight_floor: floor passed to StrictLearnableGrid.
        mode: lookup mode, one of `"softmax"` or `"falloff"`.
        sharpness: positive softness control. Larger means sharper behavior.
        falloff_kind: falloff kernel used when `mode == "falloff"`.
    """

    def __init__(
        self,
        n_points: int,
        low_init: float,
        high_init: float,
        min_spacing: float,
        dtype: torch.dtype = torch.float32,
        weight_floor: float = 1e-4,
        mode: Literal["softmax", "falloff"] = "softmax",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
    ) -> None:
        super().__init__()

        if mode == "nearest":
            raise ValueError(
                "nearest mode is not differentiable with respect to the input. "
                "Use 'softmax' or 'falloff' for LearnableLinearLut."
            )

        if mode not in {"softmax", "falloff"}:
            raise ValueError(
                f"Unknown LearnableLinearLut mode: {mode!r}. "
                "Valid modes: 'softmax', 'falloff'."
            )

        sharpness = _finite_float("sharpness", sharpness)
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")

        if falloff_kind not in FALLOFFS:
            raise ValueError(
                f"Unknown falloff_kind: {falloff_kind!r}. "
                f"Valid kinds: {sorted(FALLOFFS)}"
            )

        self.mode = mode
        self.sharpness = sharpness
        self.falloff_kind = falloff_kind

        weight_floor = _prepare_weight_floor(n_points, weight_floor)

        self.codebook = StrictLearnableGrid(
            n_points=n_points,
            low_init=low_init,
            high_init=high_init,
            min_spacing=min_spacing,
            dtype=dtype,
            weight_floor=weight_floor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a soft codebook lookup with the same shape as `x`."""
        _validate_float_tensor("x", x)

        codebook = self.codebook()
        x = x.to(codebook.device)

        if self.mode == "softmax":
            weights = _soft_codebook_weights(
                x=x,
                centers=codebook,
                sharpness=self.sharpness,
            )
        else:
            # Raw falloff weights.
            raw_weights = _falloff_weights(
                x=x,
                centers=codebook,
                sharpness=self.sharpness,
                kind=self.falloff_kind,
            )

            # Normalize when there is mass. If compact-support falloffs give zero
            # mass, fall back to the softmax kernel so the lookup stays defined.
            denom = raw_weights.sum(dim=-1, keepdim=True)
            eps = torch.finfo(raw_weights.dtype).eps

            normalized_weights = raw_weights / denom.clamp_min(eps)
            fallback_weights = _soft_codebook_weights(
                x=x,
                centers=codebook,
                sharpness=self.sharpness,
            )

            weights = torch.where(
                denom > eps,
                normalized_weights,
                fallback_weights,
            )

        # weights: x.shape + (K,)
        # codebook: (K,)
        out = (weights * codebook.to(torch.float32)).sum(dim=-1)

        return out.to(codebook.dtype)


class LearnableActivationLut(nn.Module):
    """Differentiable learnable activation LUT.

    This learns a 1-D function from `x` knot positions and `y` knot values.

    Modes:
        "linear":
            Exact piecewise-linear interpolation over `(x_i, y_i)`.

        "softmax":
            Differentiable soft assignment over `x_i`, then weighted sum of `y_i`.

        "falloff":
            Raw falloff-basis sum:

                f(x) = sum_i falloff(x; x_i, radius) * y_i

            The falloff radius is derived from the learned `x` grid and the
            same `sharpness` value used by softmax mode:

                radius = mean_gap / sharpness

    The `x` knots are constrained to be strictly increasing by
    `StrictLearnableGrid`. The `y` values are unconstrained learnable
    parameters unless you replace them with another constrained grid.

    This module is shape-agnostic and returns a tensor with the same shape as
    the input.

    Args:
        n_points: number of knots.
        low_init: initial lowest x-knot position.
        high_init: initial highest x-knot position.
        min_spacing: minimum spacing between x-knots.
        dtype_x: dtype of the learned x-knot grid.
        dtype_y: dtype of the learned y-values and returned output.
        weight_floor: floor passed to StrictLearnableGrid.
        mode: one of `"linear"`, `"softmax"`, or `"falloff"`.
            Legacy `"bilinear"` is accepted as an alias for `"linear"`.
            `"nearest"` is rejected because it is not differentiable.
        sharpness: positive softness/locality control.
        falloff_kind: falloff kernel used when `mode == "falloff"`.
        y_low_init: optional lower initialization value for y.
        y_high_init: optional upper initialization value for y.
    """

    def __init__(
        self,
        n_points: int,
        low_init: float,
        high_init: float,
        min_spacing: float,
        dtype_x: torch.dtype = torch.float32,
        dtype_y: torch.dtype = torch.float32,
        weight_floor: float = 1e-4,
        mode: Literal[
            "linear",
            "softmax",
            "falloff",
            "bilinear",
            "nearest",
        ] = "linear",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
        y_low_init: float | None = None,
        y_high_init: float | None = None,
    ) -> None:
        super().__init__()

        # Legacy alias.
        if mode == "bilinear":
            mode = "linear"

        if mode == "nearest":
            raise ValueError(
                "nearest mode is not differentiable with respect to the input. "
                "Use 'linear', 'softmax', or 'falloff' for LearnableActivationLut."
            )

        if mode not in {"linear", "softmax", "falloff"}:
            raise ValueError(
                f"Unknown LearnableActivationLut mode: {mode!r}. "
                "Valid modes: 'linear', 'softmax', 'falloff', "
                "and legacy alias 'bilinear'."
            )

        sharpness = _finite_float("sharpness", sharpness)
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")

        if falloff_kind not in FALLOFFS:
            raise ValueError(
                f"Unknown falloff_kind: {falloff_kind!r}. "
                f"Valid kinds: {sorted(FALLOFFS)}"
            )

        if not dtype_y.is_floating_point:
            raise ValueError(f"dtype_y must be floating point, got {dtype_y}")

        self.mode = mode
        self.sharpness = sharpness
        self.falloff_kind = falloff_kind

        weight_floor = _prepare_weight_floor(n_points, weight_floor)

        self.codebook_x = StrictLearnableGrid(
            n_points=n_points,
            low_init=low_init,
            high_init=high_init,
            min_spacing=min_spacing,
            dtype=dtype_x,
            weight_floor=weight_floor,
        )

        y_low = (
            low_init if y_low_init is None else _finite_float("y_low_init", y_low_init)
        )
        y_high = (
            high_init
            if y_high_init is None
            else _finite_float("y_high_init", y_high_init)
        )

        # The y-values are free parameters. They do not need to be increasing
        # unless you explicitly want a monotonic activation.
        self.codebook_y = nn.Parameter(
            torch.linspace(y_low, y_high, n_points, dtype=dtype_y)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned activation LUT with the same shape as `x`."""
        _validate_float_tensor("x", x)

        xs = self.codebook_x()
        ys = self.codebook_y

        x = x.to(xs.device)

        if self.mode == "linear":
            return _piecewise_linear_lookup(x, xs, ys)

        if self.mode == "softmax":
            weights = _soft_codebook_weights(
                x=x,
                centers=xs,
                sharpness=self.sharpness,
            )

            # weights: x.shape + (K,)
            # ys: (K,)
            out = (weights * ys.to(torch.float32)).sum(dim=-1)

            return out.to(ys.dtype)

        # self.mode == "falloff"
        #
        # Requested behavior:
        #   f(x) = sum_i falloff(x; codebook_x_i, radius) * codebook_y_i
        #
        # This is intentionally left unnormalized. It is a smooth basis-function
        # expansion centered on the learned x-knots.
        weights = _falloff_weights(
            x=x,
            centers=xs,
            sharpness=self.sharpness,
            kind=self.falloff_kind,
        )

        out = (weights * ys.to(torch.float32)).sum(dim=-1)

        return out.to(ys.dtype)


# Vector LUT helpers


def _validate_vector_input(
    name: str,
    value: torch.Tensor,
    expected_dim: int,
) -> None:
    """Validate that `value` is a floating-point `(B, C)` tensor."""
    _validate_float_tensor(name, value)

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape (batch, channels), got {tuple(value.shape)}"
        )

    if value.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_dim}, got {value.shape[-1]}"
        )


def _safe_l2_unit_and_norm(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(unit_vector, raw_norm)` in float32.

    The unit vector is safe for zero-length inputs: zero vectors produce zero
    unit vectors instead of NaNs.
    """
    value32 = value.to(torch.float32)
    norm = value32.norm(dim=-1, keepdim=True)

    eps = torch.finfo(torch.float32).eps
    safe_norm = torch.clamp_min(norm, eps)

    return value32 / safe_norm, norm


def _cosine_similarity_to_codebook(
    x: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine similarity between each input vector and each codebook vector.

    Args:
        x: input tensor shaped `(B, C)`.
        codebook: codebook tensor shaped `(K, C)`.

    Returns:
        `(sim, x_norm)` where:
          - `sim` has shape `(B, K)` and contains cosine similarities.
          - `x_norm` has shape `(B, 1)` and contains raw input L2 norms.
    """
    x_unit, x_norm = _safe_l2_unit_and_norm(x)
    c_unit, _ = _safe_l2_unit_and_norm(codebook)

    # (B, C) @ (C, K) -> (B, K)
    sim = torch.matmul(x_unit, c_unit.t())

    # Numerical safety: cosine similarity should be in [-1, 1].
    sim = sim.clamp(-1.0, 1.0)

    return sim, x_norm


def _vector_radius_from_sharpness(sharpness: float) -> float:
    """Convert a positive sharpness value into a cosine-space radius.

    Cosine similarity lives in [-1, 1]. A radius of 1.0 covers similarities
    down to 0.0. Larger radii cover more negative similarities. Smaller radii
    make the kernel more local.
    """
    radius = 1.0 / sharpness

    if not math.isfinite(radius):
        radius = 1e4

    eps = torch.finfo(torch.float32).eps
    return float(min(max(radius, eps), 1e4))


def _vector_softmax_weights(
    sim: torch.Tensor,
    sharpness: float,
) -> torch.Tensor:
    """Normalized softmax weights from cosine similarities."""
    # Cosine similarity is bounded, but large sharpness values can still create
    # large logits. Clamping keeps the softmax numerically tame.
    logits = torch.clamp(sharpness * sim, min=-1e4, max=1e4)
    return F.softmax(logits, dim=-1)


def _vector_linear_weights(
    sim: torch.Tensor,
    sharpness: float,
) -> torch.Tensor:
    """Unnormalized linear cosine-distance weights.

    weight = max(0, 1 - distance / radius)

    where distance = 1 - cosine_similarity.
    """
    radius = _vector_radius_from_sharpness(sharpness)

    distance = 1.0 - sim
    weights = (1.0 - distance / radius).clamp_min(0.0)

    return weights


def _vector_falloff_weights(
    sim: torch.Tensor,
    sharpness: float,
    kind: str,
) -> torch.Tensor:
    """Unnormalized falloff weights over cosine similarity.

    The kernel is centered at perfect similarity:

        center = 1.0

    and has radius:

        radius = 1 / sharpness

    so larger sharpness gives a narrower kernel.
    """
    if kind not in FALLOFFS:
        raise ValueError(
            f"Unknown falloff_kind: {kind!r}. Valid kinds: {sorted(FALLOFFS)}"
        )

    radius = _vector_radius_from_sharpness(sharpness)

    sim_flat = sim.reshape(-1)

    if sim_flat.numel() == 0:
        return torch.empty_like(sim, dtype=torch.float32)

    weights_flat = falloff(
        sim_flat,
        center=1.0,
        radius=radius,
        kind=kind,
    )

    return weights_flat.reshape(sim.shape)


def _normalize_vector_weights(
    weights: torch.Tensor,
    sim: torch.Tensor,
    sharpness: float,
) -> torch.Tensor:
    """Normalize weights, falling back to softmax if the weight mass is zero.

    This is especially useful for compact-support falloffs, which can produce
    zero mass when the input is far from every codebook vector.
    """
    denom = weights.sum(dim=-1, keepdim=True)
    eps = torch.finfo(weights.dtype).eps

    normalized = weights / denom.clamp_min(eps)
    fallback = _vector_softmax_weights(sim, sharpness)

    return torch.where(denom > eps, normalized, fallback)


def _restore_magnitude(
    value: torch.Tensor,
    magnitude: torch.Tensor,
) -> torch.Tensor:
    """Normalize `value` and rescale it to `magnitude`.

    `magnitude` is expected to be shaped `(B, 1)`.
    """
    value32 = value.to(torch.float32)
    value_norm = value32.norm(dim=-1, keepdim=True)

    eps = torch.finfo(torch.float32).eps
    safe_norm = torch.clamp_min(value_norm, eps)

    return (value32 / safe_norm) * magnitude


def _vector_mixture_weights(
    sim: torch.Tensor,
    mode: str,
    sharpness: float,
    falloff_kind: str,
) -> torch.Tensor:
    """Mixture weights over codebook entries from cosine similarities."""
    if mode == "softmax":
        return _vector_softmax_weights(sim, sharpness)
    if mode == "linear":
        raw_weights = _vector_linear_weights(sim, sharpness)
        return _normalize_vector_weights(raw_weights, sim, sharpness)
    raw_weights = _vector_falloff_weights(sim, sharpness, falloff_kind)
    return _normalize_vector_weights(raw_weights, sim, sharpness)


# Vector LUTs


def _validate_vector_lut_config(
    n_points: int,
    input_dim: int,
    mode: str,
    sharpness: float,
    falloff_kind: str,
    dtype: torch.dtype,
    owner: str,
) -> float:
    """Validate shared vector-LUT constructor config. Returns finite sharpness."""
    if isinstance(n_points, bool) or not isinstance(n_points, int):
        raise ValueError(f"n_points must be an int, got {n_points!r}")
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")

    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise ValueError(f"input_dim must be an int, got {input_dim!r}")
    if input_dim < 1:
        raise ValueError(f"input_dim must be at least 1, got {input_dim}")

    if mode == "nearest":
        raise ValueError(
            "nearest mode is not differentiable with respect to the input. "
            f"Use 'softmax', 'linear', or 'falloff' for {owner}."
        )

    if mode not in {"softmax", "linear", "falloff"}:
        raise ValueError(
            f"Unknown {owner} mode: {mode!r}. "
            "Valid modes: 'softmax', 'linear', 'falloff'."
        )

    sharpness = _finite_float("sharpness", sharpness)
    if sharpness <= 0.0:
        raise ValueError(f"sharpness must be positive, got {sharpness}")

    if falloff_kind not in FALLOFFS:
        raise ValueError(
            f"Unknown falloff_kind: {falloff_kind!r}. Valid kinds: {sorted(FALLOFFS)}"
        )

    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point, got {dtype}")

    return sharpness


class LearnableVectorLut(nn.Module):
    """Differentiable vector codebook lookup.

    This learns a codebook of vectors and interpolates between codebook entries
    using weights derived from cosine similarity between the input vector and
    each codebook vector.

    Input shape:
        `(B, C)`

    Output shape:
        `(B, C)`

    Modes:
        "softmax":
            Weights are `softmax(sharpness * cosine_similarity)`.

        "linear":
            Weights are linear cosine-distance weights:

                max(0, 1 - (1 - sim) / radius)

            then normalized.

        "falloff":
            Weights are generated by one of the falloff kernels centered at
            `sim == 1.0`, then normalized with a softmax fallback.

    Args:
        n_points: number of codebook vectors.
        input_dim: vector dimensionality, i.e. input `C`.
        dtype: dtype of the learned codebook and returned output.
        mode: one of `"softmax"`, `"linear"`, or `"falloff"`.
        sharpness: positive sharpness/locality control. Larger means sharper.
        falloff_kind: falloff kernel used when `mode == "falloff"`.
        preserve_magnitude: if True, output direction is normalized and then
            rescaled to match the input L2 norm.
    """

    def __init__(
        self,
        n_points: int,
        input_dim: int,
        dtype: torch.dtype = torch.float32,
        mode: Literal["softmax", "linear", "falloff"] = "softmax",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
        preserve_magnitude: bool = False,
    ) -> None:
        super().__init__()

        sharpness = _validate_vector_lut_config(
            n_points,
            input_dim,
            mode,
            sharpness,
            falloff_kind,
            dtype,
            "LearnableVectorLut",
        )

        self.input_dim = input_dim
        self.mode = mode
        self.sharpness = sharpness
        self.falloff_kind = falloff_kind
        self.preserve_magnitude = preserve_magnitude

        # Initialize codebook vectors on the unit sphere. They remain fully
        # learnable; the cosine similarity path normalizes them anyway.
        codebook = torch.randn(n_points, input_dim, dtype=torch.float32)
        codebook = F.normalize(codebook, dim=-1).to(dtype)

        self.codebook = nn.Parameter(codebook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a soft vector-codebook lookup with shape `(B, C)`."""
        _validate_vector_input("x", x, self.input_dim)

        x = x.to(self.codebook.device)
        codebook = self.codebook

        sim, x_norm = _cosine_similarity_to_codebook(x, codebook)

        weights = _vector_mixture_weights(
            sim, self.mode, self.sharpness, self.falloff_kind
        )

        # weights: (B, K)
        # codebook: (K, C)
        out = torch.matmul(weights, codebook.to(torch.float32))

        if self.preserve_magnitude:
            out = _restore_magnitude(out, x_norm)

        return out.to(codebook.dtype)


class LearnableVectorActivationLut(nn.Module):
    """Differentiable vector-valued activation LUT.

    This learns two codebooks:

      - `codebook_x`: input anchor vectors.
      - `codebook_y`: output anchor vectors.

    For an input vector `x`, weights are computed from cosine similarity
    between `x` and every `codebook_x` entry. The output is then the weighted
    sum of the matching `codebook_y` entries.

    Input shape:
        `(B, C)`

    Output shape:
        `(B, C)`

    Modes:
        "softmax":
            Weights are `softmax(sharpness * cosine_similarity)`.

        "linear":
            Weights are linear cosine-distance weights, then normalized.

        "falloff":
            Weights are generated by one of the falloff kernels centered at
            perfect cosine similarity, then normalized with a softmax fallback.

    Args:
        n_points: number of `(x, y)` vector pairs.
        input_dim: vector dimensionality, i.e. input `C`.
        dtype: dtype of learned codebooks and returned output.
        mode: one of `"softmax"`, `"linear"`, or `"falloff"`.
        sharpness: positive sharpness/locality control. Larger means sharper.
        falloff_kind: falloff kernel used when `mode == "falloff"`.
        preserve_magnitude: if True, output direction is normalized and then
            rescaled to match the input L2 norm.
    """

    def __init__(
        self,
        n_points: int,
        input_dim: int,
        dtype: torch.dtype = torch.float32,
        mode: Literal["softmax", "linear", "falloff"] = "softmax",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
        preserve_magnitude: bool = False,
    ) -> None:
        super().__init__()

        sharpness = _validate_vector_lut_config(
            n_points,
            input_dim,
            mode,
            sharpness,
            falloff_kind,
            dtype,
            "LearnableVectorActivationLut",
        )

        self.input_dim = input_dim
        self.mode = mode
        self.sharpness = sharpness
        self.falloff_kind = falloff_kind
        self.preserve_magnitude = preserve_magnitude

        # Initialize input anchors on the unit sphere.
        codebook_x = torch.randn(n_points, input_dim, dtype=torch.float32)
        codebook_x = F.normalize(codebook_x, dim=-1).to(dtype)

        # Initialize output anchors as a copy of the input anchors so the
        # initial mapping is close to an identity-like vector function.
        self.codebook_x = nn.Parameter(codebook_x.clone())
        self.codebook_y = nn.Parameter(codebook_x.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned vector activation LUT with shape `(B, C)`."""
        _validate_vector_input("x", x, self.input_dim)

        x = x.to(self.codebook_x.device)

        xs = self.codebook_x
        ys = self.codebook_y

        sim, x_norm = _cosine_similarity_to_codebook(x, xs)

        weights = _vector_mixture_weights(
            sim, self.mode, self.sharpness, self.falloff_kind
        )

        # weights: (B, K)
        # ys: (K, C)
        out = torch.matmul(weights, ys.to(torch.float32))

        if self.preserve_magnitude:
            out = _restore_magnitude(out, x_norm)

        return out.to(ys.dtype)


class ValueCenteredQuantizationLUT(nn.Module):
    """Differentiable value-centered symmetric quantization LUT.

    This module learns a shared non-negative magnitude codebook and applies it
    symmetrically around a center value.

    Let:

        d = x - center
        r = |d|

    The module learns a magnitude function:

        g(r) >= 0
        g(0) = 0

    and returns:

        f(x) = center + sign(d) * g(r)

    This guarantees, up to floating-point precision:

        x == center  -> f(x) == center
        x >  center  -> f(x) >  center
        x <  center  -> f(x) <  center

    and symmetric behavior around the center:

        f(center + d) - center = -(f(center - d) - center)

    Modes:
        ste=False:
            Fully differentiable soft quantization.

        ste=True:
            QAT-style behavior. The forward pass snaps to the nearest symmetric
            codebook value, while gradients flow through a differentiable soft
            surrogate.

    Args:
        n_points_negative: number of negative-side codebook entries, excluding
            the center. Must equal `n_points_positive` because the quantizer is
            symmetric.
        n_points_positive: number of positive-side codebook entries, excluding
            the center. Must equal `n_points_negative`.
        absolute_high_init: initial maximum absolute deviation from center.
        min_spacing: minimum spacing inside the shared magnitude codebook.
        center: scalar center value.
        dtype: dtype of the learned codebook and returned output.
        weight_floor: floor passed to StrictLearnableGrid.
        mode: soft-lookup mode for the shared magnitude LUT, one of
            `"softmax"` or `"falloff"`.
        sharpness: positive sharpness control. Larger means sharper behavior.
        falloff_kind: falloff kernel used when `mode == "falloff"`.
        learnable_center: if True, `center` becomes a learnable parameter.
        ste: if True, forward snaps to codebook values and uses a soft
            straight-through surrogate for gradients.
        strict_sign: if True, forces nonzero inputs to produce nonzero
            deviations from center. This is needed for the requirement:
                x > center -> f(x) > center
                x < center -> f(x) < center
        sign_guard: small positive guard used in soft mode to keep the sign
            strict. Set to 0 only if you do not need strict sign preservation.
    """

    def __init__(
        self,
        n_points_negative: int,
        n_points_positive: int,
        absolute_high_init: float,
        min_spacing: float,
        center: float = 0.0,
        dtype: torch.dtype = torch.float32,
        weight_floor: float = 1e-4,
        mode: Literal["softmax", "falloff"] = "softmax",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
        learnable_center: bool = False,
        ste: bool = False,
        strict_sign: bool = True,
        sign_guard: float = 1e-6,
    ) -> None:
        super().__init__()

        if isinstance(n_points_negative, bool) or not isinstance(
            n_points_negative, int
        ):
            raise ValueError(
                f"n_points_negative must be an int, got {n_points_negative!r}"
            )

        if isinstance(n_points_positive, bool) or not isinstance(
            n_points_positive, int
        ):
            raise ValueError(
                f"n_points_positive must be an int, got {n_points_positive!r}"
            )

        if n_points_negative != n_points_positive:
            raise ValueError(
                "ValueCenteredQuantizationLUT is symmetric. "
                "n_points_negative and n_points_positive must be equal. "
                f"Got {n_points_negative} and {n_points_positive}."
            )

        n_side = n_points_positive

        if n_side < 1:
            raise ValueError(
                "n_points_negative and n_points_positive must each be at least 1, "
                "because the shared magnitude codebook needs at least one positive "
                f"level besides the center. Got {n_side}."
            )

        absolute_high_init = _finite_float("absolute_high_init", absolute_high_init)
        if absolute_high_init <= 0.0:
            raise ValueError(
                f"absolute_high_init must be positive, got {absolute_high_init}"
            )

        center = _finite_float("center", center)

        sharpness = _finite_float("sharpness", sharpness)
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")

        sign_guard = _finite_float("sign_guard", sign_guard)
        if sign_guard < 0.0:
            raise ValueError(f"sign_guard must be non-negative, got {sign_guard}")

        if mode not in {"softmax", "falloff"}:
            raise ValueError(
                f"Unknown ValueCenteredQuantizationLUT mode: {mode!r}. "
                "Valid modes: 'softmax', 'falloff'."
            )

        if falloff_kind not in FALLOFFS:
            raise ValueError(
                f"Unknown falloff_kind: {falloff_kind!r}. "
                f"Valid kinds: {sorted(FALLOFFS)}"
            )

        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be floating point, got {dtype}")

        self.dtype = dtype
        self.n_side = n_side
        self.ste = bool(ste)
        self.strict_sign = bool(strict_sign)
        self.sign_guard = sign_guard

        if learnable_center:
            self.center = nn.Parameter(torch.tensor(center, dtype=dtype))
        else:
            self.register_buffer(
                "center",
                torch.tensor(center, dtype=dtype),
                persistent=True,
            )

        # Shared magnitude codebook:
        #
        #   [0, m_1, m_2, ..., m_n]
        #
        # The full centered codebook is:
        #
        #   [-m_n, ..., -m_2, -m_1, center, m_1, m_2, ..., m_n]
        #
        # Because the same magnitude LUT is used on both sides, the function is
        # symmetric around the center by construction.
        self.magnitude_lut = LearnableLinearLut(
            n_points=n_side + 1,
            low_init=0.0,
            high_init=absolute_high_init,
            min_spacing=min_spacing,
            dtype=dtype,
            weight_floor=weight_floor,
            mode=mode,
            sharpness=sharpness,
            falloff_kind=falloff_kind,
        )

        # The magnitude codebook must start at zero. The center itself is
        # handled by `self.center`, so the side LUT's zero endpoint should not
        # drift away from 0.
        with torch.no_grad():
            self.magnitude_lut.codebook.low.fill_(0.0)
        self.magnitude_lut.codebook.low.requires_grad_(False)

    @property
    def total_n_points(self) -> int:
        """Total number of centered codebook entries, including the center."""
        return 2 * self.n_side + 1

    @property
    def codebook(self) -> torch.Tensor:
        """Full centered codebook in ascending order.

        Returns:
            Tensor shaped `(total_n_points,)` containing:

                center - m_n, ..., center - m_1,
                center,
                center + m_1, ..., center + m_n
        """
        mag = self.magnitude_lut.codebook().detach().to(torch.float32)
        center = self.center.detach().to(torch.float32)

        negative_side = center - mag[1:].flip(0)
        center_point = center.reshape(1)
        positive_side = center + mag[1:]

        full = torch.cat([negative_side, center_point, positive_side])
        return full.to(self.dtype)

    def _soft_magnitude(self, r: torch.Tensor) -> torch.Tensor:
        """Differentiable non-negative magnitude response.

        Args:
            r: non-negative input magnitude, same shape as input.

        Returns:
            Non-negative magnitude `g(r)` with `g(0) == 0`.
        """
        zero = torch.zeros((), dtype=torch.float32, device=r.device)

        base = self.magnitude_lut(zero).to(torch.float32)
        raw = self.magnitude_lut(r).to(torch.float32)

        # Force the center to be exactly zero in magnitude space.
        g = raw - base

        guard = self.sign_guard

        # If strict sign preservation is required, make sure every nonzero
        # radius produces a strictly positive magnitude. Without this, soft
        # lookup responses can numerically collapse to zero near the center.
        if self.strict_sign:
            guard = max(guard, torch.finfo(torch.float32).eps)

        if guard > 0.0:
            # `guard * r` is monotonic, zero at r=0, and positive for r>0.
            # Taking the maximum preserves non-negativity and helps enforce:
            #   x > center -> f(x) > center
            #   x < center -> f(x) < center
            g = torch.maximum(g, guard * r)
        else:
            g = g.clamp_min(0.0)

        return g

    def _hard_magnitude(self, r: torch.Tensor) -> torch.Tensor:
        """Hard nearest-neighbor magnitude lookup.

        This is the forward-pass snapping operation used when `ste=True`.

        Args:
            r: non-negative input magnitude.

        Returns:
            Nearest non-negative magnitude codebook value.
        """
        centers = self.magnitude_lut.codebook().to(torch.float32)
        k = centers.numel()

        r_flat = r.reshape(-1)

        if r_flat.numel() == 0:
            return torch.empty_like(r, dtype=torch.float32)

        # Clamp for nearest-neighbor search so +inf saturates to the largest
        # magnitude codebook entry.
        r_search = torch.clamp(r_flat, min=centers[0], max=centers[-1])

        idx = torch.searchsorted(centers, r_search)
        idx = idx.clamp(0, k - 1)

        left = (idx - 1).clamp(0)
        right = idx

        d_left = torch.abs(r_search - centers[left])
        d_right = torch.abs(r_search - centers[right])

        choose_left = d_left <= d_right
        idx = torch.where(choose_left, left, right)

        # Strict sign preservation:
        #
        # If r > 0, do not allow quantization to the zero magnitude. This
        # enforces:
        #
        #   x > center -> f(x) > center
        #   x < center -> f(x) < center
        #
        # If you want classic deadzone quantization, set strict_sign=False.
        if self.strict_sign and k > 1:
            idx = torch.where(
                (idx == 0) & (r_flat > 0),
                1,
                idx,
            )

        return centers[idx].reshape(r.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return centered symmetric quantization with the same shape as `x`."""
        _validate_float_tensor("x", x)

        x = x.to(self.center.device)

        x32 = x.to(torch.float32)
        center32 = self.center.to(torch.float32)

        d = x32 - center32
        r = d.abs()

        # sign(d) gives the correct side of the center.
        #
        # At d == 0, sign(d) == 0, and the magnitude response is also zero,
        # so the output is exactly center.
        unit = torch.sign(d)

        if self.ste:
            # QAT-style forward:
            #
            #   forward value: hard snapped codebook value
            #   backward gradient: differentiable soft surrogate
            #
            # The expression:
            #
            #   hard + (soft - soft.detach())
            #
            # has the numerical value of `hard`, but the gradient of `soft`.
            hard_mag = self._hard_magnitude(r)
            hard_out = center32 + unit * hard_mag

            soft_mag = self._soft_magnitude(r)
            soft_out = center32 + unit * soft_mag

            out32 = hard_out + (soft_out - soft_out.detach())
        else:
            # Fully smooth/differentiable soft quantization.
            soft_mag = self._soft_magnitude(r)
            out32 = center32 + unit * soft_mag

        return out32.to(self.dtype)
