from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    SubtractExpression,
)


def test_expression_get_children():
    constant = ConstantExpression(4)
    variable = VariableExpression("x")
    expr = AddExpression(constant, variable)
    # expect two children for add expression
    assert len(expr.get_children()) == 2
    # when both children are present, the 0 index should be the left child
    assert expr.get_children()[0] == constant
    assert expr.evaluate({"x": 10}) == 14


def test_clone_expressions():
    constant = ConstantExpression(4)
    assert constant.value == 4
    assert constant.clone().value == 4


def test_expression_clone_root():
    a = ConstantExpression(1100)
    b = ConstantExpression(100)
    sub = SubtractExpression(a, b)
    c = ConstantExpression(300)
    add = AddExpression(sub, c)
    d = ConstantExpression(37)
    expr = AddExpression(add, d)
    assert expr.evaluate() == 1337
