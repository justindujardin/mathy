from ..tree_node import BinaryTreeNode
from ..expressions import ConstantExpression, VariableExpression, AddExpression
from ..parser import ExpressionParser


def test_parser_to_string():
    parser = ExpressionParser()
    expression = parser.parse("7 + 4x - 2")
    assert str(expression) == "7 + 4x - 2"
    # Test to make sure parens are preserved in output
    expression = parser.parse("(7 - (5 - 3)) * (32 - 7)")
    assert str(expression) == "(7 - (5 - 3)) * (32 - 7)"


def test_mult_exp_precedence():
    """should respect order of operations with factor parsing"""
    parser = ExpressionParser()
    expression = parser.parse("4x^2")
    val = expression.evaluate({"x": 2})
    # 4x^2 should evaluate to 16 with x=2
    assert val == 16

