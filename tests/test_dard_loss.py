"""DARD Phase III: the differentiable rule loss and the EMA loss balancer.

The compiled sparse path is pinned against the independent Python-loop oracle
in `nanochat/ops/references/dard_reference.py` -- values *and* gradients --
because the whole point of the log-domain formulation is that it survives
where a naive product underflows, and only a parity check against a
differently-written implementation can show that.

python -m pytest tests/test_dard_loss.py -v
"""

from __future__ import annotations

import pytest
import torch

from nanochat.models.dard.compile import compile_rules
from nanochat.models.dard.loss import DardLossBalancer, DardRuleLoss
from nanochat.models.dard.types import AmlRule, DardRuleSet
from nanochat.ops.references.dard_reference import reference_dard_loss

GAMMA = 10.0
EPS = 1e-8


def _rule_sets(i: int) -> tuple[frozenset[int], frozenset[int]]:
    """Rule `i`'s premise/conclusion: 3 disjoint tokens each, rules overlapping."""
    return (frozenset(range(i, i + 3)), frozenset(range(i + 3, i + 6)))


def make_rule_set(num_rules: int = 4, vocab: int = 12) -> DardRuleSet:
    """A compiled rule set with disjoint premises/conclusions.

    Disjointness is required for a rule to be informative at all: a rule whose
    conclusion contains its premise has a penalty sitting at its floor by
    construction, so it could never teach anything.
    """
    rules = [AmlRule(*_rule_sets(i), 1.0) for i in range(num_rules)]
    weights = torch.full((num_rules,), 1.0 / num_rules)
    return compile_rules(rules, weights, torch.ones(num_rules), vocab_size=vocab)


def make_loss(num_rules: int = 4, vocab: int = 12) -> DardRuleLoss:
    return DardRuleLoss(make_rule_set(num_rules, vocab), gamma=GAMMA, eps=EPS)


def grad_of(tensor: torch.Tensor) -> torch.Tensor:
    """`tensor.grad`, asserted present rather than Optional-checked.

    Every caller needs a real gradient -- that is the assertion's whole point --
    so failing loudly beats an `Optional` narrowed away at the use site.
    """
    grad = tensor.grad
    assert grad is not None, f"{tuple(tensor.shape)} tensor has no gradient"
    return grad


# --- Parity against the oracle -----------------------------------------------


@pytest.mark.parametrize("shape", [(2, 5, 12), (1, 1, 12), (3, 12)])
def test_when_compiled_loss_then_it_matches_the_python_oracle(shape) -> None:
    """Value parity: sparse log-domain path == explicit per-rule sum."""
    torch.manual_seed(3)
    loss_fn = make_loss()
    logits = torch.randn(*shape)
    rs = make_rule_set(4, 12)
    expected = reference_dard_loss(
        logits, rs.premise.to_dense(), rs.conclusion.to_dense(), rs.weights, GAMMA, EPS
    )
    got = loss_fn(logits)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_when_compiled_loss_backprops_then_gradients_match_the_oracle() -> None:
    """Gradient parity: the log-domain chain rule must reproduce the oracle.

    This is the assertion that the `M . log P` formulation is not just
    numerically close but analytically the same function.

    Compared at `gradient_scale=1.0`, the literal PRD formula. The default
    scale deliberately changes the gradient (see `DardRuleLoss`: the PRD's
    gradient is ~5 orders of magnitude short of cross-entropy), and a test that
    silently absorbed that factor would stop being able to detect a genuine
    change in the loss's mathematics. The scaling itself has its own test.
    """
    torch.manual_seed(4)
    loss_fn = DardRuleLoss(
        make_rule_set(4, 12), gamma=GAMMA, eps=EPS, gradient_scale=1.0
    )
    a = torch.randn(2, 4, 12, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    rs = make_rule_set(4, 12)
    loss_fn(a).backward()
    reference_dard_loss(
        b, rs.premise.to_dense(), rs.conclusion.to_dense(), rs.weights, GAMMA, EPS
    ).backward()
    assert torch.allclose(grad_of(a), grad_of(b), atol=1e-6, rtol=1e-4)


def test_when_gradient_scale_is_one_then_the_forward_is_the_literal_prd_value() -> None:
    """`gradient_scale` must change only the gradient, never the value.

    The scale is applied as `s*total - (s-1)*total.detach()`, whose value is
    `total`. If it leaked into the forward the logged penalty would stop being
    comparable across runs, which is the diagnostic this whole module relies on.
    """
    torch.manual_seed(14)
    rs = make_rule_set(4, 12)
    unscaled = DardRuleLoss(rs, gamma=GAMMA, eps=EPS, gradient_scale=1.0)
    scaled = DardRuleLoss(rs, gamma=GAMMA, eps=EPS, gradient_scale=7.0)
    logits = torch.randn(2, 3, 12)
    assert torch.allclose(unscaled(logits), scaled(logits), atol=1e-7)


def test_when_gradient_scale_is_large_then_the_gradient_is_large() -> None:
    """The correction must actually reach the gradient, by exactly the factor."""
    torch.manual_seed(15)
    rs = make_rule_set(4, 12)
    logits = torch.randn(2, 3, 12)
    norms = {}
    for scale in (1.0, 100.0):
        a = logits.clone().requires_grad_(True)
        DardRuleLoss(rs, gamma=GAMMA, eps=EPS, gradient_scale=scale)(a).backward()
        norms[scale] = grad_of(a).norm().item()
    assert norms[100.0] == pytest.approx(100.0 * norms[1.0], rel=1e-3)


def test_when_logits_are_bf16_then_the_loss_is_still_computed_in_fp32() -> None:
    """PRD §5: bf16 truncation of the violation gradient is the failure mode.

    The autocast-disabled block is what prevents it, so the assertion is that a
    bf16 input and its fp32 cast give the *same* penalty -- if the guard were
    removed, bf16 rounding of `log(P + eps)` would move the result.
    """
    torch.manual_seed(5)
    loss_fn = make_loss()
    logits = torch.randn(2, 3, 12)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        from_bf16 = loss_fn(logits.to(torch.bfloat16))
    plain = loss_fn(logits)
    assert torch.allclose(from_bf16, plain, atol=1e-5)


# --- The loss actually responds to what it should ---------------------------


def _sharp_on(tokens: tuple[int, ...], vocab: int = 12) -> torch.Tensor:
    """One position whose probability mass sits on `tokens`."""
    logits = torch.full((1, 1, vocab), -8.0)
    logits[0, 0, list(tokens)] = 8.0
    return logits


def test_when_the_model_puts_mass_on_the_conclusion_then_the_penalty_drops() -> None:
    """The gradient signal points the right way: satisfying a rule costs less.

    This is the behavioural contract of the whole feature. A penalty that did
    not fall when the model learned the rule would be a constant offset and
    DARD would be silently doing nothing.
    """
    loss_fn = make_loss(num_rules=1, vocab=12)
    # Rule 0 is `premise {0,1,2} <= conclusion {3,4,5}`, so mass on the premise
    # is the violation and mass on the conclusion is the satisfaction.
    assert loss_fn(_sharp_on((3, 4, 5))) < loss_fn(_sharp_on((0, 1, 2)))


def test_when_every_rule_is_satisfied_then_the_penalty_is_at_its_floor() -> None:
    """softplus(0)/gamma = log(2)/gamma, the minimum the hinge can reach."""
    torch.manual_seed(7)
    loss_fn = make_loss(num_rules=3, vocab=12)
    # A uniform distribution gives v_premise == v_conclusion exactly (same
    # cardinality on both sides), so every violation is 0.
    uniform = torch.zeros(1, 1, 12)
    expected = torch.log(torch.tensor(2.0)) / GAMMA
    assert loss_fn(uniform).item() == pytest.approx(expected.item(), abs=1e-5)


def test_when_no_rules_are_compiled_then_the_penalty_is_exactly_zero() -> None:
    """An empty rule set contributes 0.0, not a nan or a crash."""
    empty = compile_rules([], torch.zeros(0), torch.zeros(0), vocab_size=12)
    loss_fn = DardRuleLoss(empty, gamma=GAMMA, eps=EPS)
    assert loss_fn(torch.randn(2, 3, 12)).item() == 0.0


def test_when_a_rule_carries_more_weight_then_it_moves_the_total_more() -> None:
    """w_r must actually scale the penalty, not just ride along.

    Two rules over the same logits: rule 0 is violated, rule 1 is satisfied.
    Weighting rule 0 heavily must pull the total further from the hinge floor
    than weighting rule 1 does. Without that, the whole Phase II weighting stage
    would be decoration.
    """
    rules = [
        AmlRule(frozenset({0, 1, 2}), frozenset({3, 4, 5}), 1.0),
        AmlRule(frozenset({6, 7, 8}), frozenset({9, 10, 11}), 1.0),
    ]
    logits = _sharp_on((0, 1, 2))

    def build(w0: float) -> DardRuleLoss:
        return DardRuleLoss(
            compile_rules(
                rules, torch.tensor([w0, 1.0 - w0]), torch.ones(2), vocab_size=12
            ),
            gamma=GAMMA,
            eps=EPS,
        )

    floor = torch.log(torch.tensor(2.0)) / GAMMA
    heavy = (build(0.9)(logits) - floor).item()
    light = (build(0.1)(logits) - floor).item()
    assert heavy > light


def _rule_sets(i: int) -> tuple[frozenset[int], frozenset[int]]:
    return (frozenset(range(i, i + 3)), frozenset(range(i + 3, i + 6)))


# --- Long premises: the underflow the log domain exists for -----------------


def test_when_a_premise_is_long_then_the_product_does_not_flush_to_zero() -> None:
    """PRD §5: `prod p_i` underflows past ~5 tokens; the log domain must not.

    Each token gets probability ~1e-3, so a 12-token premise has a direct
    product of ~1e-36 -- representable in fp64, but the *gradient* through the
    naive form is what dies. The assertion is that the log-domain penalty is
    still strictly positive and still carries gradient.
    """
    vocab = 12
    premise = frozenset(range(vocab))
    conclusion = frozenset({vocab - 1})
    rules = [AmlRule(premise, conclusion, 1.0)]
    rs = compile_rules(rules, torch.ones(1), torch.ones(1), vocab_size=vocab)
    loss_fn = DardRuleLoss(rs, gamma=GAMMA, eps=EPS)

    logits = torch.full((1, 1, vocab), -10.0)
    logits.requires_grad_(True)
    logits.retain_grad()
    # Make conclusion token dominant so the order is satisfied, yet the premise
    # product is astronomically small.
    with torch.no_grad():
        logits[0, 0, vocab - 1] = 10.0
    total = loss_fn(logits)
    assert total.item() > 0.0
    total.backward()
    assert torch.isfinite(grad_of(logits)).all()
    assert grad_of(logits).abs().sum() > 0


# --- Input validation --------------------------------------------------------


def test_when_logits_vocab_disagrees_with_the_rules_then_actionable_error() -> None:
    """The single most important validation: rules for a different tokenizer."""
    loss_fn = make_loss(vocab=12)
    with pytest.raises(ValueError, match="were built for"):
        loss_fn(torch.randn(2, 3, 11))


def test_when_logits_are_1d_then_actionable_error() -> None:
    loss_fn = make_loss(vocab=12)
    with pytest.raises(ValueError, match="at least 2-D"):
        loss_fn(torch.randn(12))


def test_when_logits_are_integral_then_actionable_error() -> None:
    loss_fn = make_loss(vocab=12)
    with pytest.raises(ValueError, match="float type"):
        loss_fn(torch.zeros(1, 2, 12, dtype=torch.long))


@pytest.mark.parametrize(
    "kwargs, match",
    [({"gamma": 0.0}, "gamma must be > 0"), ({"eps": 0.0}, "eps must be > 0")],
)
def test_when_loss_params_are_invalid_then_actionable_error(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DardRuleLoss(make_rule_set(2, 12), **kwargs)


# --- The balancer ------------------------------------------------------------


def test_when_balancer_runs_then_both_terms_are_scaled_to_unit() -> None:
    """The whole point: L_CE ~ 5 and L_AML ~ 1e-3 must reach comparable scale."""
    balancer = DardLossBalancer(lambda_global=0.2, ema_decay=0.5)
    balancer.train()
    ce = torch.tensor(5.0, requires_grad=True)
    aml = torch.tensor(1e-3, requires_grad=True)
    # Feed a steady state so the EMA settles near the loss values.
    for _ in range(20):
        total, parts = balancer(ce, aml)
    assert total.item() == pytest.approx(1.0 + 0.2 * 1.0, abs=0.05)
    assert parts["ce"].item() == pytest.approx(5.0)
    assert parts["aml"].item() == pytest.approx(1e-3)


def test_when_balancer_is_used_then_both_terms_still_backprop() -> None:
    """Normalization must not detach: both losses must reach their graphs."""
    balancer = DardLossBalancer(lambda_global=0.2)
    balancer.train()
    ce = torch.tensor(2.0, requires_grad=True)
    aml = torch.tensor(0.5, requires_grad=True)
    total, _ = balancer(ce, aml)
    total.backward()
    assert ce.grad is not None and ce.grad.abs() > 0
    assert aml.grad is not None and aml.grad.abs() > 0


def test_when_the_first_step_runs_then_no_division_by_an_unseeded_ema() -> None:
    """mu starts at 0, so the warmup step must bypass normalization entirely.

    Dividing by `max(mu, eps)` with mu=0 would multiply the CE loss by 1e5.
    Both terms still pass through unweighted -- dropping the aux term here would
    leave `loss_aml.grad is None` for exactly one step, silently.
    """
    balancer = DardLossBalancer()
    balancer.train()
    ce = torch.tensor(2.0, requires_grad=True)
    aml = torch.tensor(0.5, requires_grad=True)
    total, _ = balancer(ce, aml)
    total.backward()
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    assert ce.grad is not None and ce.grad.abs() > 0
    assert aml.grad is not None and aml.grad.abs() > 0


def test_when_warmup_ends_then_the_next_step_normalizes() -> None:
    """The warmup branch is `steps <= 1` *after* the EMA update, so only the
    first call is warmup; the second call is the first normalized one.

    A warmup that ran one step too long (or too short) would flip the
    loss scale mid-run while every single-step assertion still passed:
    the unweighted step reports the raw sum, the normalized step reports
    the EMA-quotient. Pinning the boundary means the two regimes are
    distinguishable at exactly the second call.
    """
    balancer = DardLossBalancer(ema_decay=0.9, lambda_global=0.5)
    balancer.train()
    # First call: _update raises steps to 1, so the warmup branch
    # fires and the total is the unweighted sum.
    total, _ = balancer(torch.tensor(2.0), torch.tensor(0.5))
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    # Second call: _update runs first (steps -> 2), seeding the
    # EMA by one more decay step, then the normalized branch
    # divides each term by its just-updated EMA and scales the
    # aux term by lambda_global. Replicate the EMA math the
    # forward is about to apply so the expected value is exact.
    mu_ce = 0.2 * 0.9 + 2.0 * 0.1  # post-update, second call
    mu_aml = 0.05 * 0.9 + 0.5 * 0.1
    total, _ = balancer(torch.tensor(2.0), torch.tensor(0.5))
    assert total.item() == pytest.approx(
        2.0 / mu_ce + 0.5 * 0.5 / mu_aml,
        abs=1e-6,
    ), "the second call must leave the warmup branch and normalize by the EMA"


def test_when_in_eval_mode_then_the_ema_is_frozen() -> None:
    """Eval must not mutate training state: a val pass would otherwise shift mu."""
    balancer = DardLossBalancer(ema_decay=0.9)
    balancer.train()
    for _ in range(5):
        balancer(torch.tensor(1.0), torch.tensor(0.2))
    frozen = float(balancer.mu_ce)
    balancer.eval()
    balancer(torch.tensor(99.0), torch.tensor(99.0))
    assert float(balancer.mu_ce) == pytest.approx(frozen)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lambda_global": 0.0}, r"in \(0, 1\]"),
        ({"lambda_global": 1.5}, r"in \(0, 1\]"),
        ({"ema_decay": 1.0}, "ema_decay must be in"),
        ({"eps": 0.0}, "eps must be > 0"),
    ],
)
def test_when_balancer_params_are_invalid_then_actionable_error(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DardLossBalancer(**kwargs)


def test_when_balancer_state_is_saved_and_loaded_then_the_ema_resumes() -> None:
    """A resumed run must not restart the EMA from zero."""
    balancer = DardLossBalancer(ema_decay=0.9)
    balancer.train()
    for _ in range(5):
        balancer(torch.tensor(3.0), torch.tensor(0.7))
    state = balancer.state_dict()

    fresh = DardLossBalancer(ema_decay=0.9)
    fresh.load_state_dict(state)
    assert float(fresh.mu_ce) == pytest.approx(float(balancer.mu_ce))
    assert int(fresh.steps) == int(balancer.steps)
