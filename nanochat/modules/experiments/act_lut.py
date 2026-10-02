"""The `act_lut` experiment: relaxation x body parameterization of the LUT.

Counts the compiled table's parameters at a matched budget and on the model's
own codebooks, checks that the fp8 knot export resolves bit-identically, and
scores each body's fit against `relu^2`.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import AblationRow
from nanochat.models.quant.activation import (
    ACT_BODIES,
    ACT_BODY_PWL,
    ACT_BODY_SMOOTHPWL,
    SmoothPWL,
    get_activation,
)
from nanochat.models.quant.learnable_lut import (
    RELAXATION_LOGITS,
    RELAXATION_PROXIMITY,
    RELAXATIONS,
    LearnableIndexLut,
)
from nanochat.models.quant.retrofit import PRESETS, retrofit_model
from nanochat.modules.experiments.common import VARIANT_PRESET, codebook_of
from tests.conftest import build_active_tiny_gpt

#: `SmoothPWL`'s default knot-grid half-width. The class does not retain its
#: `init_range` argument as an attribute, so the default is restated here for the
#: `act_lut` fit grid, which has to score both bodies over the domain the RBF fit
#: is actually run on.
SMOOTHPWL_DEFAULT_INIT_RANGE = 1.0


def build_activation_lut(
    input_codebook: torch.Tensor,
    output_codebook: torch.Tensor,
    relaxation: str,
    act_body: str,
) -> LearnableIndexLut:
    """One `LearnableIndexLut` at the requested relaxation and body.

    The `smoothpwl` body is fitted here rather than reused, so each preset is
    measured against its own freshly fitted body instead of a fit produced by
    some earlier run.
    """
    body = None
    if act_body == ACT_BODY_SMOOTHPWL:
        body = SmoothPWL(
            knots=int(input_codebook.numel()),
            zero_pin=True,
            act_name="relu2",
        ).fit_from_callable()
    return LearnableIndexLut(
        input_codebook,
        output_codebook,
        act_name="relu2",
        relaxation=relaxation,
        smooth_body=body,
    )


#: The matched-budget K the handoff's parameter-reduction claim is stated at
#: (HANDOFF §10.2.1, "at a matched 15 floats"). Measured separately from the
#: model's own codebooks, whose K_in and K_out are not equal, so the
#: K_in x K_out vs K_in + K_out comparison has to be posed at a K where both
#: sides are stated on the same budget.
K_MATCHED_BUDGET = 15


def matched_budget_codebooks(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A `k`-level input/output codebook pair for the matched-budget count.

    The input codebook is non-negative, which is what this activation's domain
    actually is: `mlp.forward` computes `F.relu(x).square()`, so every tensor
    entering `attn.c_proj` is `>= 0`. That is what makes the pair satisfy the
    contract `LearnableIndexLut` enforces on the proximity relaxation -- the
    input level `0.0` must bake to an output codebook value of exactly `0.0`, or
    the zero anchor cannot be pinned without disagreeing with the bake, and the
    class refuses to build such a table rather than apply the pin silently.

    The output codebook is the activation's own values, which is both sorted
    (so the `bucketize` the proximity init performs is well defined) and
    contains an exact `0.0`. The parameter *count* depends only on K, not on the
    level values, so this is the right basis for the structural claim whatever
    the specific values are.
    """
    input_codebook = torch.linspace(0.0, 4.0, k, dtype=torch.float32).contiguous()
    output_codebook = get_activation("relu2")(input_codebook).contiguous()
    return input_codebook, output_codebook


def piecewise_linear(
    xs: torch.Tensor, ys: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of the polyline through `(xs, ys)` at `query`.

    This is what the frozen PWL body *is*: between knots the operation is a
    linear map, and the value at a knot is its own `y`. Endpoints are clamped
    rather than extrapolated, so a query outside the knot span takes the nearest
    end value instead of being scored against a line that does not exist there.

    `searchsorted` locates the segment and the two neighbours are lerped by
    weight. Interpolating the `x` values alone (the obvious shortcut) would
    reconstruct the *knot positions* rather than the function, and would score
    the wrong curve entirely.
    """
    idx = torch.searchsorted(xs, query, right=True).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    weight = torch.where(x1 > x0, (query - x0) / (x1 - x0), torch.zeros_like(query))
    return y0 + weight * (y1 - y0)


def run_act_lut(args: argparse.Namespace) -> list[AblationRow]:
    """Activation-LUT relaxation x body: parameter count, fp8 identity, fit.

    The table is the one compiled between two *quantized-inference* layers: the
    `mlp.c_fc` output codebook indexes it and the `attn.c_proj` input codebook
    is its value domain. Both are read off the retrofitted layer rather than
    synthesized, so the measured K is the K the model uses.

    Two of the three claims are exact and free:

    * **Parameter count.** At K_in = K_out = 15 the free-logit matrix is 225
      parameters and the proximity parameterization is K_in + K_out = 30. That
      is a structural count, not a fit result.
    * **fp8 knot identity.** `resolved_table()` reads `levels` only, so knots
      round-tripped through `float8_e4m3fn` resolve to a *bit-identical* index
      table. The saving is 1 byte per knot instead of 4 at zero behavioural
      cost. Checked by swapping the stored knots for their fp8 round trip and
      re-resolving.

    The third -- that `smoothpwl` fits `relu^2` better than `pwl` -- was measured
    in the handoff at **one** setting. It is re-measured here per preset rather
    than copied, because a number this run did not produce does not belong in a
    leaderboard meant to be auditable. If `smoothpwl` does not win at the setting
    measured, the note says so; a probe finding is a result, not a failure, and
    is not gated.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[VARIANT_PRESET])
    proj = model.transformer.h[0].attn.c_proj
    fc = model.transformer.h[0].mlp.c_fc
    # `mlp.c_fc`'s *output* quantizer feeds the activation between the two
    # layers, so its codebook is the table's index domain; `attn.c_proj`'s
    # activation (input) quantizer supplies the table's value range. Both are
    # the roles the quantized-inference path actually uses.
    input_codebook = codebook_of(fc, "out")
    output_codebook = codebook_of(proj, "act")
    k_in, k_out = int(input_codebook.numel()), int(output_codebook.numel())
    act = get_activation("relu2")
    # Both bodies are scored on one common grid over the `SmoothPWL` body's own
    # init span, because that is the domain the RBF fit is actually run over.
    # `SmoothPWL` does not retain `init_range` as an attribute, so the default
    # is restated here; it is the same constant the handoff's 1.02 / 0.48 /
    # 0.67 figures were measured on ("relu^2 over [-1, 1]", activation.py's
    # module docstring). Scoring the PWL body over the *codebook's* span instead
    # would put the two bodies on different ranges, and a `relu^2` probe over a
    # one-sided codebook scores an artefact of that range rather than a fit.
    body_range = SMOOTHPWL_DEFAULT_INIT_RANGE
    grid = torch.linspace(-body_range, body_range, 400)

    rows: list[AblationRow] = []

    # The parameter-reduction claim is stated at a *matched* budget, where
    # K_in == K_out so both relaxations are compared on the same number of
    # floats. The model's own codebooks have K_in != K_out, so that comparison is
    # posed on a matched pair instead of being asserted from the real one.
    matched_in, matched_out = matched_budget_codebooks(K_MATCHED_BUDGET)
    matched_logits = build_activation_lut(
        matched_in, matched_out, RELAXATION_LOGITS, ACT_BODY_PWL
    )
    matched_proximity = build_activation_lut(
        matched_in, matched_out, RELAXATION_PROXIMITY, ACT_BODY_PWL
    )
    matched_logits_params = sum(p.numel() for p in matched_logits.parameters())
    matched_proximity_params = sum(p.numel() for p in matched_proximity.parameters())
    # The fp8 identity is re-checked on the matched pair too, so the row
    # carries both of the exact claims it is the natural home for rather than
    # leaving one of them to be read off a per-preset row with a different K.
    with torch.no_grad():
        table_before = matched_proximity.resolved_table()
        live_knots = matched_proximity.knots.detach().clone()
        matched_proximity.knots.copy_(matched_proximity.fp8_knots())
        table_after = matched_proximity.resolved_table()
        matched_proximity.knots.copy_(live_knots)
    matched_fp8_note = (
        "fp8 knot export bit-identical to fp32: "
        f"{bool(torch.equal(table_before, table_after))}"
    )
    rows.append(
        AblationRow(
            experiment="act_lut_matched_budget",
            metric="lut_params",
            baseline=f"{RELAXATION_LOGITS}_{ACT_BODY_PWL}",
            variant=f"{RELAXATION_PROXIMITY}_{ACT_BODY_PWL}",
            value_baseline=float(matched_logits_params),
            value_variant=float(matched_proximity_params),
            delta=float(matched_proximity_params - matched_logits_params),
            better="variant"
            if matched_proximity_params < matched_logits_params
            else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"matched-budget parameter count at K_in=K_out={K_MATCHED_BUDGET}: "
                f"{matched_proximity_params} proximity parameters (K_in + K_out) vs "
                f"{matched_logits_params} for a free K_in x K_out logit matrix, a "
                f"{matched_logits_params / matched_proximity_params:.2f}x reduction. "
                f"{matched_fp8_note}. "
                "This is a structural count that depends only on K, so it is exact "
                "rather than a fit result. The model's own codebooks have "
                f"K_in={k_in} != K_out={k_out}, so this row is the matched-budget "
                "statement and the per-preset rows below are the real codebooks."
            ),
        )
    )

    for relaxation in RELAXATIONS:
        for act_body in ACT_BODIES:
            lut = build_activation_lut(
                input_codebook, output_codebook, relaxation, act_body
            )
            n_params = sum(p.numel() for p in lut.parameters())

            # fp8 identity: resolve, then swap in the fp8 round trip of the
            # stored knots and resolve again. `resolved_table()` reads levels
            # only, so the two must be bit-identical for any knot dtype.
            #
            # Only the proximity relaxation *has* knots -- the free-logit matrix
            # stores no knot positions at all, so there is nothing to export and
            # the claim does not apply to it. That is reported as such rather
            # than skipped silently, because a blanket "bit-identical: True" on
            # an arm that exported nothing would be asserting a result the run
            # never produced.
            has_knots = hasattr(lut, "knots")
            if has_knots:
                with torch.no_grad():
                    table_fp32 = lut.resolved_table()
                    live_knots = lut.knots.detach().clone()
                    lut.knots.copy_(lut.fp8_knots())
                    table_fp8 = lut.resolved_table()
                    lut.knots.copy_(live_knots)
                fp8_identical = bool(torch.equal(table_fp32, table_fp8))
                fp8_note = f"fp8 knot export bit-identical to fp32: {fp8_identical}"
            else:
                table_fp32 = lut.resolved_table()
                fp8_note = (
                    "no fp8 export: the free-logit relaxation stores no knot "
                    "positions, so the fp8 knot saving does not apply to this "
                    "preset"
                )

            # The fit is measured on the *table's* (x, y) pairs: the input
            # codebook entries are the knots, and the resolved table picks the
            # output level each knot lands on. Scoring `act(table_values)` would
            # be circular -- it would compare relu^2 against itself at the output
            # codebook -- so the curve is reconstructed from those pairs and
            # compared to the true activation over the knot span.
            knots, _order = torch.sort(input_codebook)
            y_at_knots = output_codebook[table_fp32]
            with torch.no_grad():
                table_error = float(
                    (piecewise_linear(knots, y_at_knots, grid) - act(grid)).abs().max()
                )
                if act_body == ACT_BODY_SMOOTHPWL:
                    body = SmoothPWL(
                        knots=k_in, zero_pin=True, act_name="relu2"
                    ).fit_from_callable()
                    body_error = float((body(grid) - act(grid)).abs().max())
                else:
                    # The frozen PWL body *is* that piecewise-linear curve, so
                    # both arms are scored through the same reconstruction and
                    # the comparison is like-for-like.
                    body_error = table_error

            rows.append(
                AblationRow(
                    experiment=f"act_lut_{relaxation}_{act_body}",
                    metric="lut_params",
                    baseline=f"{RELAXATION_LOGITS}_{ACT_BODY_PWL}",
                    variant=f"{relaxation}_{act_body}",
                    value_baseline=float(k_in * k_out),
                    value_variant=float(n_params),
                    delta=float(n_params - k_in * k_out),
                    better="variant" if n_params < k_in * k_out else "baseline",
                    seeds=list(range(args.seeds)),
                    n_seeds=args.seeds,
                    notes=(
                        f"relaxation={relaxation} body={act_body}: {n_params} table "
                        f"parameters vs {k_in * k_out} for a free K_in x K_out logit "
                        f"matrix at K_in={k_in}, K_out={k_out} (proximity is K_in + "
                        f"K_out = {k_in + k_out}). {fp8_note}. Measured this run at "
                        f"K={k_in}: body "
                        f"max abs error on relu^2 over a {grid.numel()}-point grid "
                        f"{body_error:.6g}, resolved-table error {table_error:.6g}. "
                        "These are re-measured per preset, not copied from the "
                        "handoff, whose figures came from one setting. No model runs."
                    ),
                )
            )
    return rows
