"""
Security tests for the calculator tool's restricted AST evaluator
(nanochat/engine.py). Formulas come from model output: they must compute
ordinary arithmetic/string.count() and reject everything else.

python -m pytest tests/test_calculator.py -v
"""

import pytest

from nanochat.modules.engine import safe_eval_expression, use_calculator


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("2+2*3", 8),
        ("(1+2)/4", 0.75),
        ("7//2", 3),
        ("-5+10", 5),
        ("1.5*2", 3.0),
        ("1,000+1", 1001),
        ("'banana'.count('an')", 2),
        ('"abcabc".count("b")', 2),
    ],
)
def test_when_valid_formula_then_evaluates(formula: str, expected) -> None:
    assert use_calculator(formula) == expected


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os')",
        "open('/etc/passwd')",
        "(1).__class__",
        "eval('2+2')",
        "1**2",
        "1 < 2",
        "x",
        "lambda: 1",
        "[x for x in range(3)]",
        "globals()",
        "'a'.strip()",
        "",
    ],
)
def test_when_hostile_or_unsupported_formula_then_rejected(formula: str) -> None:
    assert use_calculator(formula) is None


def test_when_division_by_zero_then_rejected() -> None:
    assert use_calculator("1/0") is None


def test_when_expression_uses_only_whitelisted_nodes_then_ok() -> None:
    assert safe_eval_expression("2*(3+4)") == 14


def test_when_disallowed_node_then_raises() -> None:
    with pytest.raises(ValueError):
        safe_eval_expression("2 ** 3")
    with pytest.raises(ValueError):
        safe_eval_expression("__builtins__")
