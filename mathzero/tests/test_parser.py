from ..core.tokenizer import Token
from ..core.tree import BinaryTreeNode
from ..core.expressions import ConstantExpression, VariableExpression, AddExpression
from ..core.parser import ExpressionParser


def test_parser_to_string():
    parser = ExpressionParser()
    # expression = parser.parse("7 + 4x - 2")
    # assert str(expression) == "7 + 4x - 2"
    # Test to make sure parens are preserved in output
    expression = parser.parse("(7 - (5 - 3)) * (32 - 7)")
    assert str(expression) == "(7 - (5 - 3)) * (32 - 7)"


def test_tokenizer():
    parser = ExpressionParser()
    tokens = parser.tokenize("(7x^2 - (5x - 3x)) * (32y - 7y)")

    features = [t.to_feature() for t in tokens]
    assert len(features) > 0

    tokens_out = [Token.from_feature(f) for f in features]
    assert len(tokens) == len(tokens_out)
    for i, t in enumerate(tokens_out):
        assert t.value == tokens[i].value
        assert t.type == tokens[i].type


def test_mult_exp_precedence():
    """should respect order of operations with factor parsing"""
    parser = ExpressionParser()
    expression = parser.parse("4x^2")
    val = expression.evaluate({"x": 2})
    # 4x^2 should evaluate to 16 with x=2
    assert val == 16

    expression = parser.parse("7 * 10 * 6x * 3x + 5x")
    assert expression is not None
