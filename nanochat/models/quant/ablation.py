# todo: Split into many different files because this file is too big.

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
      init, the same trick as `nanochat.models.quant.codebook`): the domain is always
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
    module its own optimizer group (see `nanochat.models.quant.optimizer`).

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


def _linear_interpolation_basis(x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Hat-function basis of the linear interpolant on the knot grid `xs`.

    Returns `[x.numel(), xs.numel()]` with `basis[i, k]` equal to the weight of
    knot `k` in the piecewise-linear interpolant at `x[i]`. Every row sums to 1,
    so `lstsq(basis @ ys)` is a partition-of-unity least-squares fit rather than
    an unconstrained regression.
    """
    n = x.numel()
    k = xs.numel()
    flat_x = x.reshape(-1, 1)
    left = torch.searchsorted(xs, flat_x.contiguous()).clamp(1, k - 1) - 1
    x0 = xs[left]
    x1 = xs[left + 1]
    span = (x1 - x0).clamp_min(torch.finfo(xs.dtype).tiny)
    t = ((flat_x - x0) / span).clamp(0.0, 1.0)
    basis = torch.zeros(n, k, dtype=xs.dtype, device=xs.device)
    rows = torch.arange(n, device=xs.device)
    basis[rows, left.reshape(-1)] = 1.0 - t.reshape(-1)
    basis[rows, (left + 1).reshape(-1)] += t.reshape(-1)
    return basis


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

    @classmethod
    def from_callable(
        cls,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        low: float,
        high: float,
        n_points: int = 16,
        min_spacing: float | None = None,
        mode: Literal["linear", "softmax", "falloff"] = "linear",
        sharpness: float = 1.0,
        falloff_kind: Literal["gaussian", "cosine", "quartic"] = "cosine",
        fit: Literal["exact", "lsq"] = "exact",
    ) -> "LearnableActivationLut":
        """Build a LUT that reproduces `act_fn` at its knots.

        The x-grid is a `StrictLearnableGrid` spanning `[low, high]`; `y_i` is
        seeded with `act_fn(x_i)`, so in `fit="exact"` + `mode="linear"` the
        piecewise-linear interpolant **equals** `act_fn` at every knot and only
        approximates it between them. `fit="lsq"` instead fits `y` to `act_fn`
        sampled on a dense grid, which is the better choice for a coarse
        `n_points`: it minimises the error *between* knots rather than only
        preserving the knot values.

        Honest framing: with `K=8` knots this is a piecewise-linear function,
        not `relu^2`. It is exact at the knots and approximate between them,
        and because it is a free `nn.Parameter` it can train past the closed
        form within the step budget. It is a strict superset of the baked
        table, not an exact drop-in for the closed form.

        Args:
            act_fn: elementwise activation on a 1-D floating-point tensor.
            low, high: domain endpoints of the x-grid.
            n_points: number of knots.
            min_spacing: minimum x spacing. Defaults to `(high-low)/(n-1)` times
                a small factor, i.e. just below uniform so the grid starts
                strictly increasing without wasting the domain.
            fit: `"exact"` seeds `y` from `act_fn` at the knots; `"lsq"` solves a
                least-squares fit over a dense sample of the same `act_fn`.
        """
        low = _finite_float("low", low)
        high = _finite_float("high", high)
        if not low < high:
            raise ValueError(f"require low < high, got low={low}, high={high}")
        if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
            raise ValueError(f"n_points must be an integer >= 2, got {n_points!r}")
        if fit not in ("exact", "lsq"):
            raise ValueError(f"fit must be 'exact' or 'lsq', got {fit!r}")

        if min_spacing is None:
            # Strictly below uniform so the grid is initially strictly
            # increasing without letting the slack swallow the domain.
            min_spacing = (high - low) / (n_points - 1) * 0.5
        min_spacing = _finite_float("min_spacing", min_spacing)
        if min_spacing <= 0.0:
            raise ValueError(f"min_spacing must be positive, got {min_spacing}")

        module = cls(
            n_points=n_points,
            low_init=low,
            high_init=high,
            min_spacing=min_spacing,
            mode=mode,
            sharpness=sharpness,
            falloff_kind=falloff_kind,
        )

        with torch.no_grad():
            xs = module.codebook_x().to(torch.float32)
            if fit == "exact":
                ys = act_fn(xs).to(torch.float32)
            else:
                # Least squares against a dense sample: the interpolant matches
                # the function in a least-squares sense over the whole domain,
                # not just at the knots, which is what a coarse grid needs.
                dense = torch.linspace(low, high, max(256, n_points * 32))
                target = act_fn(dense.to(torch.float32))
                basis = _linear_interpolation_basis(dense, xs)
                solution = torch.linalg.lstsq(basis, target.unsqueeze(-1)).solution
                ys = solution.squeeze(-1)
            module.codebook_y.copy_(ys.to(module.codebook_y.dtype))
        return module

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
