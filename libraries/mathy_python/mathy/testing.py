import json
from pathlib import Path
from typing import Type

from .core.parser import ExpressionParser
from .core.rule import BaseRule
from .util import compare_expression_string_values, compare_expression_values
from .rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)


def get_rule_tests(name):
    """Load a set of JSON rule test assertions.

    # Arguments
    name (str): The name of the test JSON file to open, e.g. "commutative_property"

    # Returns
    (dict): A dictionary with "valid" and "invalid" keys that contain pairs of 
    expected inputs and outputs.
    """
    rule_file = (
        Path(__file__).parent.parent / "tests" / "rules" / "{}.json".format(name)
    )
    if not rule_file.is_file() is True:
        raise ValueError(f"does not exist: {rule_file}")
    with open(rule_file, "r") as file:
        return json.load(file)


def init_rule_for_test(example: dict, rule_class: Type[BaseRule]) -> BaseRule:
    """Initialize a given rule_class from a test example.

    This handles optionally passing the test example constructor arguments
    to the Rule.

    # Arguments:
    example (dict): The example assertion loaded from a call to `get_rule_tests`
    rule_class (Type[BaseRule]): The 

    # Returns
    (BaseRule): The rule instance.
    """
    if "args" not in example:
        rule = rule_class()
    else:
        rule = rule_class(**example["args"])  # type: ignore
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
            target = ex["target"]
            nodes = rule.find_nodes(expression)
            targets = [n.raw for n in nodes]
            node = [n for n in nodes if n.raw == target]
            targets = "\n".join(targets)
            assert len(node) > 0, f"could not find target: {target}. targets are:\n{targets}"
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
        actual = str(after).strip()
        expected = ex["output"]
        assert actual == expected, f"Expected '{actual}' to be '{expected}'"

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
