from typing import Dict
import numpy as np
from ..core.expressions import MathExpression, VariableExpression
from ..core.parser import ExpressionParser
from ..helpers import (
    get_terms,
    terms_are_like,
    load_rule_tests,
    compare_expression_values,
    compare_expression_string_values,
)
from ..rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
    VariableMultiplyRule,
)


def init_rule_for_test(example, rule_class):
    if "args" not in example:
        rule = rule_class()
    else:
        rule = rule_class(**example["args"])
    return rule


def run_rule_tests(name, rule_class, callback=None):
    """Load and assert about the transformations and validity of rules
    based on given input examples.

    When debugging a problem it can be useful to provide a "callback" function
    and add a `"debug": true` value to the example in the rules json file you
    want to debug. Then you set a breakpoint and step out of your callback function
    into the parsing/evaluation of the debug example.
    """
    tests = load_rule_tests(name)
    parser = ExpressionParser()
    for ex in tests["valid"]:
        # Trigger the debug callback so the user can step over into the useful stuff
        if callback is not None:
            callback(ex)
        rule = init_rule_for_test(ex, rule_class)
        expression = parser.parse(ex["input"]).clone()
        before = expression.clone()
        print(ex)
        if "target" in ex:
            nodes = rule.find_nodes(expression)
            targets = [n.raw for n in nodes]
            node = [n for n in nodes if n.raw == ex["target"]]
            assert len(node) > 0, f"could not find target node. targets are: {targets}"
            node = node[0]
        else:
            node = rule.find_node(expression)

        if node is None:
            assert "expected to find node but did not for" == str(expression)
        change = rule.apply_to(node)
        after = change.result.get_root()
        # Compare the values of the in-memory expressions output from the rule
        compare_expression_values(before, after)
        # Parse the output strings to new expressions, and compare the values
        compare_expression_string_values(str(before), str(after))
        assert str(after).strip() == ex["output"]

    for ex in tests["invalid"]:
        # Trigger the debug callback so the user can step over into the useful stuff
        if callback is not None:
            callback(ex)
        rule = init_rule_for_test(ex, rule_class)
        expression = parser.parse(ex["input"]).clone()
        node = rule.find_node(expression)
        if node is not None:
            raise ValueError(
                "expected not to find a node, but found: {}".format(str(node))
            )


def test_associative_property():
    def debug(ex):
        pass

    run_rule_tests("associative_property", AssociativeSwapRule, debug)


def test_commutative_property():
    def debug(ex):
        pass

    run_rule_tests("commutative_property", CommutativeSwapRule, debug)


def test_constants_simplify():
    def debug(ex):
        pass

    run_rule_tests("constants_simplify", ConstantsSimplifyRule, debug)


def test_distributive_factor_out():
    def debug(ex):
        pass

    run_rule_tests("distributive_factor_out", DistributiveFactorOutRule, debug)


def test_distributive_multiply_across():
    def debug(ex):
        pass

    run_rule_tests("distributive_multiply_across", DistributiveMultiplyRule, debug)


def test_variable_multiply():
    def debug(ex):
        pass

    run_rule_tests("variable_multiply", VariableMultiplyRule, debug)


def test_rule_can_apply_to():
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


def test_like_terms_compare():
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
