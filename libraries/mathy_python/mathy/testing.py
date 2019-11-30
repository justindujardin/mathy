import json
from pathlib import Path
from typing import Type

from .core.parser import ExpressionParser
from .core.rule import BaseRule
from .helpers import compare_expression_string_values, compare_expression_values
from .rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)


def get_rule_tests(name):
    rule_file = (
        Path(__file__).parent.parent / "tests" / "rules" / "{}.json".format(name)
    )
    if not rule_file.is_file() is True:
        raise ValueError(f"does not exist: {rule_file}")
    with open(rule_file, "r") as file:
        return json.load(file)


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
    tests = get_rule_tests(name)
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


def get_test_from_class(class_type: Type[BaseRule]) -> str:
    if class_type == AssociativeSwapRule:
        return get_rule_tests("associative_property")
    if class_type == CommutativeSwapRule:
        return get_rule_tests("commutative_property")
    if class_type == ConstantsSimplifyRule:
        return get_rule_tests("constants_simplify")
    if class_type == DistributiveFactorOutRule:
        return get_rule_tests("distributive_factor_out")
    if class_type == DistributiveMultiplyRule:
        return get_rule_tests("distributive_multiply_across")
    if class_type == VariableMultiplyRule:
        return get_rule_tests("variable_multiply")
    raise ValueError(f"unknown class-to-tests mapping for {class_type}")
