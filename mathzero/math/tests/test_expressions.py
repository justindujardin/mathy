from ..tree_node import BinaryTreeNode
from ..expressions import ConstantExpression, VariableExpression, AddExpression


def test_expression_get_children():
    constant = ConstantExpression(4)
    variable = VariableExpression("x")
    expr = AddExpression(constant, variable)
    # expect two children for add expression
    assert len(expr.getChildren()) == 2
    # when both children are present, the 0 index should be the left child
    assert expr.getChildren()[0] == constant

    assert expr.evaluate({'x': 10}) == 14
