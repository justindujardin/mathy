from ..tree_node import BinaryTreeNode
from ..expressions import ConstantExpression, VariableExpression, AddExpression, DivideExpression
from ..properties.commutative import CommutativeSwapRule


def test_expression_to_string():
    expr = AddExpression(ConstantExpression(4), ConstantExpression(17))
    assert str(expr) == '4 + 17'

def test_commutative_property():
    left = ConstantExpression(4)
    right = ConstantExpression(17)
    expr = AddExpression(left, right)
    rule = CommutativeSwapRule()

    # can find the root-level nodes
    nodes = rule.findNodes(expr)
    assert len(nodes) == 1

    # This expression is commutative compatible
    assert rule.canApplyTo(expr) == True

    # Applying swaps the left/right position of the Const nodes in the Add expression
    result = rule.applyTo(expr).node
    assert result.left.value == right.value
    assert result.right.value == left.value


def test_commutative_property_cannot_apply():
    left = ConstantExpression(4)
    right = ConstantExpression(17)
    expr = DivideExpression(left, right)
    rule = CommutativeSwapRule()
    # This expression is NOT commutative compatible because 4 / 3 != 3 / 4
    assert rule.canApplyTo(expr) == False
    # Nope
    assert len(rule.findNodes(expr)) == 0
