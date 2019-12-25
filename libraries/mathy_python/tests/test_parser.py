from ..mathy.core.tokenizer import Token
from ..mathy.core.tree import BinaryTreeNode
from ..mathy.core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
)
from ..mathy.core.parser import ExpressionParser


def test_parser_to_string():
    parser = ExpressionParser()
    expects = [
        {
            "input": "(-2.257893300159429e+16h^2 * v) * j^4",
            "output": "(-2.257893300159429e + 16h^2 * v) * j^4",
        },
        {"input": "1f + 98i + 3f + 14t", "output": "1f + 98i + 3f + 14t"},
        {"input": "4x * p^(1 + 3) * 12x^2", "output": "4x * p^(1 + 3) * 12x^2"},
        {"input": "(5 * 3) * (32 / 7)", "output": "(5 * 3) * 32 / 7"},
        {"input": "7 - 5 * 3 * (2^7)", "output": "7 - 5 * 3 * 2^7"},
        {"input": "(8x^2 * 9b) * 7", "output": "(8x^2 * 9b) * 7"},
        {"input": "(8 * 9b) * 7", "output": "(8 * 9b) * 7"},
        {"input": "7 - (5 * 3) * (32 / 7)", "output": "7 - (5 * 3) * 32 / 7"},
        {"input": "7 - (5 - 3) * (32 - 7)", "output": "7 - (5 - 3) * (32 - 7)"},
        {"input": "(7 - (5 * 3)) * (32 - 7)", "output": "(7 - 5 * 3) * (32 - 7)"},
    ]
    # Test to make sure parens are preserved in output when they are meaningful
    for expect in expects:
        expression = parser.parse(expect["input"])
        out_str = str(expression)
        assert out_str == expect["output"]


def test_mult_exp_precedence():
    """should respect order of operations with factor parsing"""
    parser = ExpressionParser()
    expression = parser.parse("4x^2")
    val = expression.evaluate({"x": 2})
    # 4x^2 should evaluate to 16 with x=2
    assert val == 16

    expression = parser.parse("7 * 10 * 6x * 3x + 5x")
    assert expression is not None
