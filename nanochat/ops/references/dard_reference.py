"""Pure-PyTorch oracle for the DARD order-violation loss.

The ground truth the compiled path must reproduce, written the way the PRD's
equations read rather than the way a fused kernel would compute them. Ground
truth is a plain Python loop over rules and tokens: a tensor expression that
reuses the same `matmul` the implementation does would share the bug it is
supposed to catch.
"""

from __future__ import annotations

import torch


def validate_dard_inputs(logits: torch.Tensor, vocab_size: int) -> None:
    """Fail-fast boundary validation for the DARD loss.

    Args:
        logits: `(..., V)` logits. Must be >= 2-D so the leading dims are
            unambiguously batch/position and the last is vocabulary.
        vocab_size: the vocabulary the compiled rules were built against.
    """
    if logits.dim() < 2:
        raise ValueError(
            f"logits must be at least 2-D (..., vocab), got shape {tuple(logits.shape)}"
        )
    if logits.size(-1) != vocab_size:
        raise ValueError(
            f"logits have vocabulary width {logits.size(-1)}, but the compiled "
            f"rules were built for {vocab_size}; recompile the rules against "
            "this tokenizer's vocabulary"
        )
    if not logits.is_floating_point():
        raise ValueError(f"logits must be a float type, got {logits.dtype}")


def reference_dard_loss(
    logits: torch.Tensor,
    premise: torch.Tensor,
    conclusion: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    """Weighted mean softplus hinge on the premise/conclusion product terms.

    For each rule `r`, with `log v_r = M[r] . log(P + eps)`:

        penalty_r = softplus(gamma * (exp(log v_premise_r) - exp(log v_conclusion_r))) / gamma
        loss      = mean_r( w_r * penalty_r )

    The `eps` inside the `log` is what keeps a zero-probability token from
    producing `-inf`: a premise containing a token the model assigned
    probability 0 would otherwise make `v_premise` 0 and the rule
    vacuously satisfied. Flooring `log(P)` at `log(eps)` instead makes such a
    token maximally *unsatisfying*, which is the behaviour the PRD asks for
    (PRD §5, "Product T-Norm").
    """
    validate_dard_inputs(logits, premise.shape[1])
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + eps)

    rules: list[torch.Tensor] = []
    for r in range(premise.shape[0]):
        # Explicit sums, not `premise[r] @ log_probs`: this loop is the oracle, so it must not go through the same reduction the implementation uses.
        log_v_premise = log_probs[..., sorted(_row(premise, r))].sum(dim=-1)
        log_v_conclusion = log_probs[..., sorted(_row(conclusion, r))].sum(dim=-1)
        violation = torch.exp(log_v_premise) - torch.exp(log_v_conclusion)
        rules.append(torch.nn.functional.softplus(gamma * violation).mean() / gamma)
    if not rules:
        return torch.zeros((), dtype=logits.dtype)
    stacked = torch.stack(rules)
    return (stacked * weights).sum()


def _row(matrix: torch.Tensor, index: int) -> list[int]:
    """Column indices with a non-zero value in row `index` of a dense matrix."""
    dense = (
        matrix[index].to_dense() if matrix.layout != torch.strided else matrix[index]
    )
    return [int(i) for i in torch.nonzero(dense, as_tuple=False).flatten().tolist()]


__all__ = ["reference_dard_loss", "validate_dard_inputs"]
