"""Learnable activation bodies for the fused LUT stack (D7).

`SmoothPWL` is the radial-basis activation from `ablation/simple.py`, promoted
to a real module. It is an *alternative* body for the elementwise op between two
quantized layers, selected by `--lcqat-act-body`.

Why it exists. The shipped body is a piecewise-linear map through a frozen
`K_in -> K_out` index table, which is exact only at the knots and linear
between. `SmoothPWL` replaces the interpolation with a radial-basis map over a
learnable knot grid:

    out = sum_j softmax_j( 1 / ((x - knot_j)^2 + eps) * scale ) * (w_j * x + b_j)

`index`, `weights` and `bias` are all free `nn.Parameter` -- not constrained, not
tied to the input codebook. The measured basis for that (HANDOFF §10.2.3): at a
matched 15 floats the RBF form fits `relu^2` to a max abs error of 0.48 against
the piecewise-linear path's 1.02, and 0.67 with the zero-anchor bypass attached.
An RBF basis with free knots, free slopes and free intercepts is a strictly
richer family than a fixed grid, and that measurement is evidence *against*
narrowing it.

The zero anchor. `relu^2` is exactly 0 at `x = 0`, and LC-QAT's input codebook has
exactly one level at FP32 0.0 (index `m_neg`). A radial-basis convex combination
has no reason to pass through the origin -- measured at -4.47e-08 before the fix,
which breaks the SparseProp structural-zero contract at the exact junction of
the two methods. So one knot is pinned at 0.0 with its weight and bias forced
such that the emitted value is exactly 0.0. This is `knot-and-pin`, chosen over
a `where`-style bypass because it keeps the operation a pure convex combination
and therefore survives export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanochat.models.quant.lut import get_activation

#: Floor on the inverse-square distance. Without it an input sitting exactly on a
#: knot divides by zero. Matches `ablation/simple.py`.
RBF_EPS = 1e-6

#: Selector scale in units of the median knot spacing squared. The unscaled form
#: saturates in FP32 at realistic K: at K=15 the gap is ~0.14, so the nearest
#: knot scores ~770 against ~48 for its neighbour, a softmax argument gap of
#: ~700, and everything off-diagonal underflows to exactly 0 -- the table then
#: trains nothing while looking healthy. At 1.0 the nearest knot scores ~16
#: against ~2 (`exp(14)`, comfortably representable), so every knot keeps a
#: live gradient. Same reasoning as `PROXIMITY_SCALE_GRID_UNITS`, and measured
#: rather than assumed.
#:
#: This governs *training* only. The initializer deliberately does NOT use it
#: (see `FIT_SHARPNESS_REFERENCE`): fitting needs a near-delta basis, and
#: conflating the two costs ~30x in fit error. Measured on relu^2 over [-1,1]
#: with 5 basis functions: 0.025 at the unscaled reference vs 0.99 at the
#: training scale. The two regimes want genuinely different selectors, so the
#: initializer carries its own constant rather than borrowing this one.
RBF_SCALE_GRID_UNITS = 1.0

#: Grid-relative sharpness the *initializer* fits at, i.e. the training scale
#: omitted. A near-delta basis (weight -> 1/eps at the nearest knot, 0
#: elsewhere) is a piecewise-linear interpolator with a smooth seam, which is
#: exactly what an activation fit wants. Gradients flow to the knots during the
#: fit because Adam moves the knots off their initial on-grid positions in the
#: first step, so `diff` is no longer 0 at the nearest knot.
FIT_SHARPNESS_REFERENCE = 0.0

#: Activation bodies `--lcqat-act-body` can select.
ACT_BODY_PWL = "pwl"
ACT_BODY_SMOOTHPWL = "smoothpwl"
ACT_BODIES = (ACT_BODY_PWL, ACT_BODY_SMOOTHPWL)


class SmoothPWL(nn.Module):
    """Radial-basis activation with free knots, free slopes and free intercepts.

    Args:
        knots: number of RBF basis functions. Also the number of free
            parameters per group, so the storage is `3 * knots` floats.
        eps: floor on the inverse-square distance.
        init_range: half-width of the knot grid's initial span. Defaults to 1.0,
            matching `ablation/simple.py`'s `linspace(-1, 1)`. The knots are
            free parameters, so a badly chosen span only costs training time --
            it does not pin the result.
        init_slope: initial value of every `weights` entry. 1.0 makes the module
            start as `x` plus a learned offset, which is the natural starting
            point for an activation.
        init_bias: initial value of every `bias` entry.
        zero_pin: force the emitted value at `x == 0` to exactly 0.0. Required
            for any activation that is zero-preserving at the origin (`relu2`,
            `silu`, `gelu`, `tanh`); refuse to construct without it for those,
            because the un-pinned form returns -4.47e-08 and would break the
            SparseProp contract.
        act_name: which registered activation the fit targets. Only used to
            check the zero-preserving precondition and by `fit_from_callable`.
    """

    def __init__(
        self,
        knots: int = 10,
        eps: float = RBF_EPS,
        init_range: float = 1.0,
        init_slope: float = 1.0,
        init_bias: float = 0.0,
        zero_pin: bool = True,
        act_name: str = "relu2",
    ) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        if not zero_pin and is_zero_preserving(act_name):
            raise ValueError(
                f"{act_name} is exactly 0 at x=0, so an un-pinned SmoothPWL "
                "emits a small nonzero value there (measured: -4.47e-08) and "
                "breaks the SparseProp structural-zero contract. Pass "
                "zero_pin=True; the pin costs nothing and survives export."
            )
        self.knots_count = int(knots)
        self.eps = float(eps)
        self.act_name = act_name
        self.zero_pin = bool(zero_pin)

        # All three free. Constraining any of them would narrow a family that
        # measured strictly better than the fixed-grid alternative, with no
        # measured reason to constrain (HANDOFF §10.2.3).
        self.index = nn.Parameter(
            torch.linspace(-init_range, init_range, self.knots_count)
        )
        self.weights = nn.Parameter(torch.full((self.knots_count,), float(init_slope)))
        self.bias = nn.Parameter(torch.full((self.knots_count,), float(init_bias)))

        if self.zero_pin:
            # The knot nearest the origin is the one whose basis would
            # otherwise carry x through the intercept term at x=0. Pinning its
            # bias to exactly -0.0 is a no-op numerically, so the exact zero
            # comes from pinning the *output*: at x=0 the basis is dominated by
            # the knot at 0, and `w_j * 0 + b_j` = b_j = 0. The pin therefore
            # holds the knot at exactly 0.0 AND its bias at exactly 0.0, which
            # makes the emitted value at x=0 exactly 0.0 while every other x
            # remains free.
            pinned = int(torch.argmin(self.index.detach().abs()).item())
            mask = torch.zeros(self.knots_count, dtype=torch.bool)
            mask[pinned] = True
            self.register_buffer("pin_mask", mask, persistent=True)

    def fit_from_callable(
        self, act_fn=None, *, exact: bool = False, steps: int = 2000, lr: float = 0.05
    ) -> "SmoothPWL":
        """Initialize the basis to approximate the target activation.

        `act_fn` defaults to the registered `act_name`. With `exact=True` the
        fit is a short Adam run on the knot positions and coefficients against
        the target sampled over the knot span; with `exact=False` the knots stay
        on their uniform init and only the coefficients are fitted, which is
        cheaper but cannot fix a bad span.

        This is an initializer, not the shipped behaviour: the parameters stay
        free afterwards, which is the whole point. Adam is deliberately used
        rather than L-BFGS or a closed form, because the objective is not
        convex in the knots and the step count is bounded so it cannot run away
        on a pathological target.

        Not decorated `@torch.no_grad()`: the `exact` branch is a real fit and
        needs the graph. Only the grid/target construction and the least-squares
        branch are wrapped, which is where no_grad belongs.
        """
        fn = get_activation(self.act_name) if act_fn is None else act_fn
        with torch.no_grad():
            span = float(self.index.detach().abs().max())
        # A denser sample grid than the RBF basis: the fit is scored between the
        # knots too, where the body is only an approximation, and 64 points for 5
        # basis functions would leave most of the span unconstrained.
        grid = torch.linspace(-span, span, 400)
        with torch.no_grad():
            target = fn(grid)

        def _evaluate(x: torch.Tensor) -> torch.Tensor:
            """The body at the *fit* sharpness.

            Deliberately not `self(x)`: training runs the selector at
            `RBF_SCALE_GRID_UNITS`, but the fit wants the near-delta basis at
            `FIT_SHARPNESS_REFERENCE`. Evaluating the training body during the
            fit optimizes the wrong objective -- measured 0.99 max abs error on
            relu^2 against 0.025 for the reference basis.
            """
            index = self._pinned_index()
            diff = x.unsqueeze(-1) - index
            weights = 1.0 / (diff.pow(2) + self.eps)
            if FIT_SHARPNESS_REFERENCE != 0.0:
                scale = self._spacing()
                if scale > 0.0:
                    weights = weights * (FIT_SHARPNESS_REFERENCE * scale**2)
            basis = torch.softmax(weights, dim=-1)
            return (self.weights * basis).sum(dim=-1) * x + (
                self._pinned_bias() * basis
            ).sum(dim=-1)

        if exact:
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = (_evaluate(grid) - target).square().mean()
                loss.backward()
                opt.step()
            return self
        with torch.no_grad():
            # Least squares on the coefficients with the basis held fixed. The
            # map is linear in (weights, bias): out = sum_j basis_j * (w_j * x
            # + b_j) = (basis * x) @ w + basis @ b, so the design matrix has one
            # row per sample and one column per (weight, bias) pair. Built from
            # two 2-D operands with an explicit `cat` on dim 1 -- broadcasting a
            # `cat` operand promotes only that one operand to a higher rank, and
            # `cat` then rejects the rank mismatch.
            basis = torch.softmax(
                1.0 / ((grid.reshape(-1, 1) - self._pinned_index()).pow(2) + self.eps),
                dim=-1,
            )  # (n_grid, n_knots)
            wx = basis * grid.reshape(-1, 1)  # (n_grid, n_knots)
            design = torch.cat([wx, basis], dim=1)  # (n_grid, 2 * n_knots)
            sol = torch.linalg.lstsq(design, target.reshape(-1, 1)).solution.reshape(-1)
            self.weights.copy_(sol[: self.knots_count])
            self.bias.copy_(sol[self.knots_count :])
        return self

    def _basis(self, x: torch.Tensor, sharpness: float | None = None) -> torch.Tensor:
        """Softmax assignment of each `x` to each knot, shape `(*x.shape, knots)`.

        `sharpness` is the grid-relative selector scale, defaulting to the
        training scale. Pass `FIT_SHARPNESS_REFERENCE` to get the near-delta
        basis the initializer fits at -- see that constant for why the two
        regimes must not share a value.
        """
        diff = x.unsqueeze(-1) - self.index
        weights = 1.0 / (diff.pow(2) + self.eps)
        if sharpness is None:
            sharpness = RBF_SCALE_GRID_UNITS
        if sharpness != 0.0:
            scale = self._spacing()
            if scale > 0.0:
                weights = weights * (sharpness * scale**2)
        return torch.softmax(weights, dim=-1)

    def _spacing(self) -> float:
        """Median local knot gap, read detached.

        A median, so an unevenly spaced knot grid is not dominated by its wide
        side; and detached, because backpropagating through it would turn the
        knot gradient into a one-hot scatter that is zero for every knot but
        one -- exactly the freeze the scale exists to prevent.
        """
        with torch.no_grad():
            gaps = (self.index[1:] - self.index[:-1]).abs()
            return float(gaps.median())

    def _pinned_bias(self) -> torch.Tensor:
        """`bias` with the pinned knot forced to exactly 0.0."""
        if not self.zero_pin:
            return self.bias
        return torch.where(
            self.pin_mask.to(self.bias.device),
            torch.zeros((), dtype=self.bias.dtype, device=self.bias.device),
            self.bias,
        )

    def _pinned_index(self) -> torch.Tensor:
        """`index` with the pinned knot held at exactly 0.0."""
        if not self.zero_pin:
            return self.index
        return torch.where(
            self.pin_mask.to(self.index.device),
            torch.zeros((), dtype=self.index.dtype, device=self.index.device),
            self.index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the activation on any shape; the knot axis is appended."""
        index = self._pinned_index()
        diff = x.unsqueeze(-1) - index
        weights = 1.0 / (diff.pow(2) + self.eps)
        scale = self._spacing()
        if scale > 0.0:
            weights = weights * (RBF_SCALE_GRID_UNITS * scale**2)
        basis = torch.softmax(weights, dim=-1)
        return (self.weights * basis).sum(dim=-1) * x + (
            self._pinned_bias() * basis
        ).sum(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"knots={self.knots_count}, eps={self.eps}, "
            f"zero_pin={self.zero_pin}, act={self.act_name}"
        )


def is_zero_preserving(act_name: str) -> bool:
    """Whether `act_name` is exactly 0 at x=0.

    Probed numerically rather than hard-coded, so a new registry entry cannot
    silently get the wrong answer: `sigmoid` is the one registered activation
    that is not, and it is the case the zero pin would silently corrupt.
    """
    fn = get_activation(act_name)
    return float(fn(torch.tensor(0.0))) == 0.0


__all__ = [
    "ACT_BODIES",
    "ACT_BODY_PWL",
    "ACT_BODY_SMOOTHPWL",
    "RBF_EPS",
    "RBF_SCALE_GRID_UNITS",
    "SmoothPWL",
    "is_zero_preserving",
]
