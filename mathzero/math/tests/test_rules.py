from ..tree_node import BinaryTreeNode
from ..parser import ExpressionParser
from ..expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    DivideExpression,
)
from ..rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)


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
    assert str(out) == '7 * (1 + 1)'


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

