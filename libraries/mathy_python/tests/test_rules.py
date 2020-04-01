from typing import Dict
import numpy as np
import json
from pathlib import Path
from mathy.core.expressions import MathExpression, VariableExpression
from mathy.core.parser import ExpressionParser
from mathy.util import (
    get_terms,
    terms_are_like,
    compare_expression_values,
    compare_expression_string_values,
)
from mathy.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
    VariableMultiplyRule,
)

from mathy.testing import init_rule_for_test, run_rule_tests


def test_rules_associative_property():
    def debug(ex):
        pass

    run_rule_tests("associative_property", AssociativeSwapRule, debug)


def test_rules_commutative_property():
    def debug(ex):
        pass

    run_rule_tests("commutative_property", CommutativeSwapRule, debug)


def test_rules_constants_simplify():
    def debug(ex):
        pass

    run_rule_tests("constants_simplify", ConstantsSimplifyRule, debug)


def test_rules_distributive_factor_out():
    def debug(ex):
        pass

    run_rule_tests("distributive_factor_out", DistributiveFactorOutRule, debug)


def test_rules_distributive_multiply_across():
    def debug(ex):
        pass

    run_rule_tests("distributive_multiply_across", DistributiveMultiplyRule, debug)


def test_rules_variable_multiply():
    def debug(ex):
        pass

    run_rule_tests("variable_multiply", VariableMultiplyRule, debug)


def test_rules_rule_can_apply_to():
    parser = ExpressionParser()
    expression = parser.parse("7 + 4x - 2")

    available_actions = [
        CommutativeSwapRule(),
        DistributiveFactorOutRule(),
        DistributiveMultiplyRule(),
        AssociativeSwapRule(),
    ]
    for action in available_actions:
        assert type(action.can_apply_to(expression)) == bool


def test_rules_like_terms_compare():
    parser = ExpressionParser()
    expr = parser.parse("10 + (7x + 6x)")
    terms = get_terms(expr)
    assert len(terms) == 3
    assert not terms_are_like(terms[0], terms[1])
    assert terms_are_like(terms[1], terms[2])

    expr = parser.parse("10 + 7x + 6")
    terms = get_terms(expr)
    assert len(terms) == 3
    assert not terms_are_like(terms[0], terms[1])
    assert terms_are_like(terms[0], terms[2])

    expr = parser.parse("6x + 6 * 5")
    terms = get_terms(expr)
    assert len(terms) == 2
    assert not terms_are_like(terms[0], terms[1])

    expr = parser.parse("360y^1")
    terms = get_terms(expr)
    assert len(terms) == 1

    expr = parser.parse("4z")
    terms = get_terms(expr)
    assert len(terms) == 1
