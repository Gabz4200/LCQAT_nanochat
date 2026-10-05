"""DARD: Differentiable Algebraic Rule Distillation.

Offline AML rule mining compiles discrete semilattice order relations into
static sparse indicator matrices; this package holds the compiled form and the
differentiable loss that reads it.
"""

from nanochat.models.dard.atomize import atomize_shard
from nanochat.models.dard.compile import compile_rules, mine_rules
from nanochat.models.dard.loss import DardLossBalancer, DardRuleLoss
from nanochat.models.dard.shard import shard_corpus
from nanochat.models.dard.types import AmlRule, DardRuleSet, ShardRules
from nanochat.models.dard.weighting import composite_rule_weights

__all__ = [
    "AmlRule",
    "DardLossBalancer",
    "DardRuleLoss",
    "DardRuleSet",
    "ShardRules",
    "atomize_shard",
    "compile_rules",
    "composite_rule_weights",
    "mine_rules",
    "shard_corpus",
]
