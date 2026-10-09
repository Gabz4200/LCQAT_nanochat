"""Phase III: the differentiable DARD loss and the dynamic loss balancer.

Two objects, and the split is deliberate:

* `DardRuleLoss` reads a compiled `DardRuleSet` and turns a logit distribution
  into one scalar -- the order-violation penalty. It holds no state that
  depends on the step.
* `DardLossBalancer` owns the two loss *scales*, not the losses. It is what
  keeps `L_AML` from being crushed by `L_CE` (PRD §3.3.3).

**Why the `autocast(enabled=False)` block is not optional.** The whole loss
runs in fp32 even under a bf16 autocast, for three independent reasons, each
of which is a documented failure mode in PRD §5:

1. `log(P + eps)` with a `P` containing entries near `eps = 1e-8`. bf16 carries
   8 mantissa bits, so `1e-8 + p` rounds to `p` outright for any `p > ~1e-6`,
   and the `+eps` guard silently stops guarding.
2. The compiled `weights` buffer. 1e-6 is below fp8 E5M2's smallest subnormal
   (2^-16 ~ 1.5e-5, arXiv:2209.05433 Table 1), so the floor would flush to
   zero -- zeroing exactly the tail rules it exists to keep alive.
3. The softplus hinge `softplus(gamma * violation)` with `gamma = 10`: bf16
   resolution near the saturating tail is coarse enough to lose the gradient
   the hinge exists to provide.

Consistent with mixed-precision practice (arXiv:1710.03740 §3.3), the matmuls
would be *allowed* to run reduced-precision for throughput -- but not the
log-domain reductions or the accumulation, which is what this block enforces.

**The two-matrix trick.** A rule is enforced by the *difference* of two
log-domain sums over disjoint token sets. Writing them as one matrix and one
product would be marginally cheaper but would require materialising a
signed dense matrix; the sparse sum is `O(nnz)` either way and keeps both
indicator matrices binary, which is what makes them cacheable and
exportable. (This is the CPU SpMM argument of arXiv:2209.05433 / Liu et al.
2014 applied to a rule set rather than a weight matrix.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.dard.types import DardRuleSet
from nanochat.ops.references.dard_reference import validate_dard_inputs


class _ScaleGradient(torch.autograd.Function):
    """Identity forward, `scale` backward.

    The penalty's *value* must stay the PRD formula -- it is the logged
    diagnostic and has to stay comparable across runs -- while its *gradient*
    needs correcting by up to ~1e8. Expressing that as `s*x - (s-1)*x.detach()`
    inside the graph forms `s*x`, which overflows fp32 at that magnitude; this
    does the rescaling on the incoming gradient instead, where the product is
    formed with whatever magnitude the gradient actually has.

    A custom Function rather than the arithmetic identity because the identity
    is unrepresentable in fp32 at the scales involved, and silently producing
    `inf`/`nan` here would be invisible in the loss curve.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0] * ctx.scale, None


class DardRuleLoss(nn.Module):
    """Weighted softplus penalty on premise/conclusion order violations.

    Args:
        rule_set: the compiled rules. Its buffers are registered here, so this
            module's own `state_dict` does carry the whole rule set.
        gamma: hinge temperature. Larger makes the penalty more nearly a hard
            `max(0, violation)`; smaller smooths the gradient around zero.
        eps: floor inside `log(P + eps)`, see `reference_dard_loss`.
        gradient_scale: multiplies the penalty's gradient, restoring it to the
            order of the cross-entropy gradient. See the class-level note below;
            the default is derived from the rule set when left as `None`.

    **The penalty's gradient is structurally tiny, and `gradient_scale` is the
    correction.** The PRD's penalty differentiates a product t-norm, so

        d L / d logit  ~  sigmoid(gamma * violation) * v(premise)

    and `v` is a *product of probabilities*, which decays geometrically in the
    rule's arity (measured: k=1 -> 5e-2, k=4 -> 4e-5, k=8 -> 2e-10). Measured on
    a real 4-layer model with 3-token rules:

        CE grad norm    5.0e-01
        DARD grad norm  9.4e-07      ratio 1.9e-06

    The PRD's EMA balancer cannot fix this. It rescales the two *loss values*,
    and both are divided by a scalar, so the *gradient ratio* is unchanged --
    measured at 16.9% loss share but a 5e-07 gradient share. The penalty would
    be present in every log line and absent from every update, which is the
    "enabled but inert" failure this project has already hit twice.

    So the correction is applied to the gradient directly, and its default is
    measured rather than guessed: the mean `v(premise)` of the rule set, which
    is the constant the gradient is proportional to. That makes `DardRuleLoss`
    contribute a gradient comparable to cross-entropy at
    `lambda_global = 1`, which is the only reading under which
    "the proportion of gradient norm assigned to rule enforcement" means
    anything. Set `gradient_scale=1.0` for the literal PRD formula.

    **Not** part of the model/engine state dict. A DARD loss module is an
    optimizer-side attachment, not a trainable subtree, so `engine.state_dict()`
    does not contain it -- see `scripts/_train/loop.py`, which writes the rule
    set and the balancer EMA into the checkpoint explicitly through
    `dard_state` / `load_dard_state`. Rebuilding the rule set by re-mining on
    resume would silently enforce *different* rules, since mining depends on
    the batch it sees.
    """

    def __init__(
        self,
        rule_set: DardRuleSet,
        gamma: float = 10.0,
        eps: float = 1e-8,
        gradient_scale: float | None = None,
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if gradient_scale is not None and gradient_scale <= 0.0:
            raise ValueError(f"gradient_scale must be > 0, got {gradient_scale}")
        self.gamma = float(gamma)
        self.eps = float(eps)
        # Sparse CSR buffers move with `.to(device)` and stay sparse, which is the whole reason the compiled form is CSR rather than dense.
        self.register_buffer("premise", rule_set.premise)
        self.register_buffer("conclusion", rule_set.conclusion)
        self.register_buffer("weights", rule_set.weights)
        self.register_buffer("frequency", rule_set.frequency)
        self.vocab_size = rule_set.vocab_size
        # The calibration constant, computed once from the rule set: the mean number of premise tokens per rule. The penalty's gradient is proportional to v(premise), and v over k tokens is ~p^k, so 1/arity is the right order of magnitude to restore it. `None` means "derive it"; an explicit float pins it (1.0 = the literal PRD formula).
        self.gradient_scale = (
            self._derive_gradient_scale(rule_set)
            if gradient_scale is None
            else float(gradient_scale)
        )

    @staticmethod
    def _derive_gradient_scale(rule_set: DardRuleSet) -> float:
        """Calibrate the gradient scale from the rule set's own geometry.

        The penalty's gradient is proportional to `v(premise)`, the product
        t-norm of the premise's tokens. For a premise of `k` tokens that is
        roughly `p^k` with `p` the mean per-token probability, so the
        correction is `1 / E[v]`.

        `p` is estimated as `1/V` -- the mean probability of a token under a
        near-uniform distribution, which is what an untrained model emits and
        therefore the conservative (largest-scale) estimate. That gives

            scale = V ** mean_premise_arity

        and for the measured rule sets (V=32768, mean premise arity 1.29) that
        is ~1.3e4, against a measured shortfall of ~6e5 between the DARD and CE
        gradients. Premise *arity* alone is not enough to derive the scale: the
        real rule sets have 1-token premises (dedup collapses longer ones) yet
        still need a large correction, because `v` is a product of
        probabilities and not of counts.
        """
        if rule_set.num_rules == 0:
            return 1.0
        mean_premise = rule_set.premise.values().numel() / rule_set.num_rules
        return float(rule_set.vocab_size**mean_premise)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean weighted penalty over rules, for `logits` of shape `(..., V)`.

        The returned value is the PRD penalty unchanged -- so `loss` reads the
        same and stays comparable across runs -- but its *gradient* is
        multiplied by `gradient_scale`. That asymmetry is deliberate: the
        forward value is a readable diagnostic, while the gradient is what
        actually trains, and the two need different scales. See the class
        docstring for why the PRD's scale is wrong for the latter.

        Returns a scalar. With no rules compiled the penalty is 0.0, which keeps
        the loop free of a special case when `--no-dard` is combined with an
        empty rule artifact.
        """
        validate_dard_inputs(logits, self.vocab_size)
        if self.weights.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=logits.device)

        # fp32 for the whole block; see the module docstring. `device_type` is taken from the logits so the guard is valid under cpu, cuda and mps rather than assuming one.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            flat = logits.float().reshape(-1, logits.size(-1))
            log_probs = torch.log(torch.softmax(flat, dim=-1) + self.eps)
            v_premise = torch.sparse.mm(self.premise, log_probs.t()).t()
            v_conclusion = torch.sparse.mm(self.conclusion, log_probs.t()).t()
            violation = torch.exp(v_premise) - torch.exp(v_conclusion)
            penalty = F.softplus(self.gamma * violation) / self.gamma
            total = (penalty.mean(dim=0) * self.weights).sum()
            if self.gradient_scale == 1.0:
                return total
            return _ScaleGradient.apply(total, self.gradient_scale)


class DardLossBalancer(nn.Module):
    """EMA-normalised dynamic balance between `L_CE` and `L_AML` (PRD §3.3.3).

    `L_total = L_CE / max(mu_CE, delta) + lambda * L_AML / max(mu_AML, delta)`

    The EMA is over the loss *values*, not their gradients, so it costs one
    scalar update per step and needs no extra backward. That is a deliberate
    departure from GradNorm (arXiv:1711.02257), whose `w_i` come from the
    gradient norms at a shared trunk -- an extra backward through the shared
    layers per step. The PRD specifies the EMA form, and for a two-term
    auxiliary loss it is the cheaper of the two; GradNorm's `alpha` restoring
    force is what the EMA does not provide, and is not claimed here.

    `mu` is updated from the *unbalanced* loss before it is used as the divisor,
    which is what makes the ratio `~1` by construction rather than only in
    expectation.
    """

    def __init__(
        self,
        lambda_global: float = 0.20,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        warmup: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < lambda_global <= 1.0:
            raise ValueError(f"lambda_global must be in (0, 1], got {lambda_global}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lambda_global = float(lambda_global)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        # Until the EMA has seen enough steps it sits at its initial value and the first steps would be divided by an arbitrary scale. `warmup` instead lets the un-normalized loss through while `steps == 0`, which is the AutoLoss-style ramp the PRD's §7 mitigation asks for: start on cross-entropy, ramp in rule enforcement.
        self.warmup = warmup
        self.register_buffer("mu_ce", torch.zeros(()))
        self.register_buffer("mu_aml", torch.zeros(()))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update(self, loss_ce: torch.Tensor, loss_aml: torch.Tensor) -> None:
        self.mu_ce.mul_(self.ema_decay).add_(loss_ce.detach(), alpha=1 - self.ema_decay)
        self.mu_aml.mul_(self.ema_decay).add_(
            loss_aml.detach(), alpha=1 - self.ema_decay
        )
        self.steps += 1

    def forward(
        self, loss_ce: torch.Tensor, loss_aml: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Balanced total loss, plus the two normalised terms for logging.

        Returns:
            `(total, parts)` where `parts["ce"]`, `parts["aml"]` and
            `parts["raw_aml"]` are detached scalars. The raw term is logged
            because after normalization the *reported* number is ~1 always, so
            a rule set that silently stopped enforcing anything would be
            invisible in the loss curve alone.
        """
        if self.training:
            self._update(loss_ce, loss_aml)
        detached_ce = loss_ce.detach()
        detached_aml = loss_aml.detach()
        if self.warmup and int(self.steps) <= 1:
            # First step only: mu is still 0, so dividing by max(mu, eps) would multiply the CE loss by 1e5. The aux term is *unweighted* rather than absent -- returning `loss_ce` alone would leave `loss_aml.grad is None` for that step, i.e. the rule penalty would contribute no gradient exactly once, silently.
            return loss_ce + loss_aml, {
                "ce": detached_ce,
                "aml": detached_aml,
                "raw_aml": detached_aml,
            }
        norm_ce = loss_ce / torch.clamp(self.mu_ce, min=self.eps)
        norm_aml = loss_aml / torch.clamp(self.mu_aml, min=self.eps)
        total = norm_ce + self.lambda_global * norm_aml
        return total, {
            "ce": detached_ce,
            "aml": detached_aml,
            "raw_aml": detached_aml,
        }


DARD_STATE_VERSION = 1


def dard_state(loss_fn: "DardRuleLoss", balancer: "DardLossBalancer") -> dict:
    """The DARD half of a training checkpoint: the rule set plus the EMA.

    Version-stamped like the rule-set artifact and for the same reason. The
    buffers here are what make a resumed run enforce the *same* rules: re-mining
    depends on whichever batch the resume sees, so without this a resume would
    quietly train against a different objective than the one it was measuring.

    The EMA is included because losing it costs more than it looks: `mu`
    restarts at 0, the balancer takes its warmup branch, and the first steps
    run unbalanced. One step of drift, small -- but it is avoidable drift, and
    the state is three scalars.
    """
    return {
        "version": DARD_STATE_VERSION,
        "loss": loss_fn.state_dict(),
        "balancer": balancer.state_dict(),
        "gamma": loss_fn.gamma,
        "eps": loss_fn.eps,
        "vocab_size": loss_fn.vocab_size,
    }


def load_dard_state(state: dict) -> tuple["DardRuleLoss", "DardLossBalancer"]:
    """Rebuild `(loss_fn, balancer)` from a `dard_state` payload.

    Raises:
        ValueError: on a version mismatch or a missing field. A stale payload is
            rejected rather than loaded, for the reason `RULE_SET_VERSION` and
            `NOISE_MAP_VERSION` exist.
    """
    missing = {"version", "loss", "balancer", "vocab_size"} - set(state)
    if missing:
        raise ValueError(f"DARD state is missing fields: {sorted(missing)}")
    version = state["version"]
    if version != DARD_STATE_VERSION:
        raise ValueError(
            f"DARD checkpoint state has version {version}, this build expects "
            f"{DARD_STATE_VERSION}; the rule weighting or penalty definition "
            "changed, so the saved rules mean something different"
        )

    # Rebuild the module at the *payload's* shapes, then let load_state_dict overwrite the values. A 0-row placeholder cannot accept a real rule set: `load_state_dict` checks shapes and rejects [0, V] <- [96, V].
    weights = state["loss"].get("weights")
    rows = int(weights.shape[0]) if weights is not None else 0
    vocab = int(state["vocab_size"])
    empty_csr = lambda: torch.sparse_csr_tensor(  # noqa: E731
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.float32),
        size=(rows, vocab),
    )
    template = DardRuleSet(
        premise=empty_csr(),
        conclusion=empty_csr(),
        weights=torch.zeros(rows, dtype=torch.float32),
        frequency=torch.zeros(rows, dtype=torch.float32),
        vocab_size=vocab,
    )
    loss_fn = DardRuleLoss(
        template,
        gamma=float(state.get("gamma", 10.0)),
        eps=float(state.get("eps", 1e-8)),
    )
    balancer = DardLossBalancer()
    # Assigned rather than `load_state_dict`: copying a populated CSR into a placeholder one goes through `copy_`, which requires an identical nnz count and rejects the transfer. Reading the buffers out of the payload is also the honest operation here -- these *are* the rule set, not a delta to be merged into one.
    saved = state["loss"]
    for name in ("premise", "conclusion", "weights", "frequency"):
        setattr(loss_fn, name, saved[name].to(device="cpu"))
    balancer.load_state_dict(state["balancer"])
    return loss_fn, balancer


__all__ = [
    "DARD_STATE_VERSION",
    "DardLossBalancer",
    "DardRuleLoss",
    "dard_state",
    "load_dard_state",
]
