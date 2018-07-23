from ..tree_node import BinaryTreeNode
from ..expressions import ConstantExpression, VariableExpression, AddExpression
from ..parser import ExpressionParser


def test_mult_exp_precedence():
    """should respect order of operations with factor parsing"""
    parser = ExpressionParser()
    expression = parser.parse("4x^2")
    val = expression.evaluate({"x": 2})
    # 4x^2 should evaluate to 16 with x=2
    assert val == 16

