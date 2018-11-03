from ..core.tree import BinaryTreeNode
from ..core.layout import TreeLayout, TidierExtreme, TreeMeasurement
from ..core.parser import ExpressionParser
from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    MultiplyExpression,
    DivideExpression,
)
from ..core.util import getTerms, termsAreLike, load_rule_tests
from ..core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
    VariableMultiplyRule,
)

# TODO: Incorporate competency evaluations in training? Adjust hyper params/problems when certain competencies are met?
exam_combine_like_terms = [
    ("10 + (7x + 6x)", "10 + 13x"),
    ("6x + 6 * 5 - 2x", "4x + 30"),
    ("(x * 14 + 7x) + 2", "21x + 2"),
    ("4x + 7x + 2", "11x + 2"),
    ("6x + 120x", "126x"),
    ("3x + 72x", "75x"),
]

exam_simplify_complex_terms = [
    ("60 * 6y", "360y"),
    ("4x * 7x", "28x^2"),
    ("(x * 14 + 7x) + 2", "21x + 2"),
    ("4x + 7x + 2", "11x + 2"),
    ("6x + 120x", "126x"),
    ("3x + 72x", "75x"),
    ("4x^(2^2)", "4x^4"),
    ("(4x^2)^2", "16x^4"),
]


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
    rule = rule_class()
    has_valid_debug = sum([1 if "debug" in e else 0 for e in tests["valid"]]) > 0
    has_invalid_debug = sum([1 if "debug" in e else 0 for e in tests["invalid"]]) > 0
    has_debug = has_invalid_debug or has_valid_debug
    for ex in tests["valid"]:
        # Skip over non-debug examples if there are any for easier debugging.
        if has_debug and "debug" not in ex:
            continue
        # Trigger the debug callback so the user can step over into the useful stuff
        if callback is not None:
            callback(ex)
        expression = parser.parse(ex["input"])
        print(ex)
        node = rule.findNode(expression)
        assert node is not None
        change = rule.applyTo(node)
        assert str(change.end.getRoot()).strip() == ex["output"]
    for ex in tests["invalid"]:
        # Skip over non-debug examples if there are any for easier debugging.
        if has_debug and "debug" not in ex:
            continue
        # Trigger the debug callback so the user can step over into the useful stuff
        if callback is not None:
            callback(ex)
        expression = parser.parse(ex["input"])
        node = rule.findNode(expression)
        assert node is None


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
        assert type(action.canApplyTo(expression)) == bool


def test_like_terms_compare():
    parser = ExpressionParser()
    expr = parser.parse("10 + (7x + 6x)")
    terms = getTerms(expr)
    assert len(terms) == 3
    assert not termsAreLike(terms[0], terms[1])
    assert termsAreLike(terms[1], terms[2])

    expr = parser.parse("10 + 7x + 6")
    terms = getTerms(expr)
    assert len(terms) == 3
    assert not termsAreLike(terms[0], terms[1])
    assert termsAreLike(terms[0], terms[2])

    expr = parser.parse("6x + 6 * 5")
    terms = getTerms(expr)
    assert len(terms) == 2
    assert not termsAreLike(terms[0], terms[1])

    expr = parser.parse("360y^1")
    terms = getTerms(expr)
    assert len(terms) == 1

    expr = parser.parse("4z")
    terms = getTerms(expr)
    assert len(terms) == 1
