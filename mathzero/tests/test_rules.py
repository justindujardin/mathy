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
from ..core.util import getTerms, termsAreLike
from ..core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
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


def test_commutative_property():
    left = ConstantExpression(4)
    right = ConstantExpression(17)
    expr = AddExpression(left, right)
    assert str(expr) == "4 + 17"
    rule = CommutativeSwapRule()

    # can find the root-level nodes
    nodes = rule.findNodes(expr)
    assert len(nodes) == 1

    # This expression is commutative compatible
    assert rule.canApplyTo(expr) == True

    # Applying swaps the left/right position of the Const nodes in the Add expression
    result = rule.applyTo(expr).end.getRoot()
    assert result.left.value == right.value
    assert result.right.value == left.value
    assert str(expr) == "17 + 4"


def test_commutative_property_cannot_apply():
    left = ConstantExpression(4)
    right = ConstantExpression(17)
    expr = DivideExpression(left, right)
    rule = CommutativeSwapRule()
    # This expression is NOT commutative compatible because 4 / 3 != 3 / 4
    assert rule.canApplyTo(expr) == False
    # Nope
    assert len(rule.findNodes(expr)) == 0


def test_commutative_property_truncate():
    parser = ExpressionParser()
    expr = parser.parse("(7 + x) + 2")
    rule = CommutativeSwapRule()

    change = rule.applyTo(expr)
    assert str(change.end) == "2 + (7 + x)"


def test_constants_simplify_rule():
    parser = ExpressionParser()
    expectations = [("7 + 4", 11), ("1 - 2", -1), ("4 / 2", 2), ("5 * 5", 25)]
    for input, output in expectations:
        expression = parser.parse(input)
        rule = ConstantsSimplifyRule()
        assert rule.canApplyTo(expression) == True
        assert rule.applyTo(expression).end.getRoot().value == output


def test_distributive_factoring():
    parser = ExpressionParser()
    expression = parser.parse("7 + 7")
    rule = DistributiveFactorOutRule()
    assert rule.canApplyTo(expression) == True
    out = rule.applyTo(expression).end.getRoot()
    assert str(out) == "7 * (1 + 1)"


def test_distributive_factoring_with_variables():
    parser = ExpressionParser()
    expression = parser.parse("14x + 7x")
    rule = DistributiveFactorOutRule()
    assert rule.canApplyTo(expression) == True
    out = rule.applyTo(expression).end.getRoot()
    assert str(out) == "7x * (2 + 1)"


def test_distributive_factoring_factors():
    pass
    parser = ExpressionParser()
    expression = parser.parse("4 + (z + 4)")
    rule = DistributiveFactorOutRule()
    assert rule.canApplyTo(expression) == False


def test_common_properties_can_apply_to():
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


def test_associative_swap():
    parser = ExpressionParser()
    exp = parser.parse("(2 + x) + 9")
    rule = AssociativeSwapRule()
    nodes = rule.findNodes(exp)
    assert len(nodes) == 1
    change = rule.applyTo(nodes[0])
    assert str(change.end.getRoot()).strip() == "2 + (x + 9)"


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
