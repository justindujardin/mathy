import pytest

from mathy import (
    AbsExpression,
    AddExpression,
    BinaryExpression,
    ConstantExpression,
    DivideExpression,
    EqualExpression,
    ExpressionParser,
    FunctionExpression,
    MathExpression,
    MultiplyExpression,
    NegateExpression,
    PowerExpression,
    SgnExpression,
    SubtractExpression,
    UnaryExpression,
    VariableExpression,
)


def test_expressions_get_children():
    constant = ConstantExpression(4)
    variable = VariableExpression("x")
    expr = AddExpression(constant, variable)
    # expect two children for add expression
    assert len(expr.get_children()) == 2
    # when both children are present, the 0 index should be the left child
    assert expr.get_children()[0] == constant
    assert expr.evaluate({"x": 10}) == 14


def test_expressions_type_id_abstract():
    expr = MathExpression()
    with pytest.raises(NotImplementedError):
        cool_var = expr.type_id


def test_expressions_evaluate_abstract():
    expr = MathExpression()
    with pytest.raises(NotImplementedError):
        expr.evaluate()


def test_expressions_add_class():
    expr = VariableExpression("x")
    expr.add_class("as_string")
    expr.add_class(["many_classes", "as_list"])
    math_ml = expr.to_math_ml()
    assert "as_string" in math_ml
    assert "many_classes" in math_ml
    assert "as_list" in math_ml


def test_expressions_clear_classes():
    expr = VariableExpression("x")
    expr.add_class("as_string")
    math_ml = expr.to_math_ml()
    assert "as_string" in expr.classes
    expr.clear_classes()
    assert "as_string" not in expr.classes


@pytest.mark.parametrize(
    "node_instance",
    [
        AddExpression(ConstantExpression(3), ConstantExpression(1)),
        SubtractExpression(ConstantExpression(3), ConstantExpression(3)),
        MultiplyExpression(ConstantExpression(3), ConstantExpression(3)),
        DivideExpression(ConstantExpression(3), ConstantExpression(3)),
        NegateExpression(ConstantExpression(3)),
        ConstantExpression(3),
        SgnExpression(ConstantExpression(-1)),
        SgnExpression(ConstantExpression(0)),
        SgnExpression(ConstantExpression(1)),
        AbsExpression(ConstantExpression(-1)),
        PowerExpression(VariableExpression("x"), ConstantExpression(3)),
    ],
)
def test_expressions_common_properties_and_methods(node_instance: MathExpression):
    assert node_instance.type_id is not None
    assert node_instance.name is not None
    assert str(node_instance) != ""
    node_instance.evaluate({"x": 2})


@pytest.mark.parametrize(
    "node_instance",
    [
        BinaryExpression(ConstantExpression(1), ConstantExpression(1)),
        UnaryExpression(ConstantExpression(1)),
        FunctionExpression(ConstantExpression(1)),
    ],
)
def test_expressions_abstract_properties_and_methods(node_instance: MathExpression):
    with pytest.raises(NotImplementedError):
        node_instance.evaluate()


def test_expressions_equality_evaluate_error():
    one = VariableExpression("x")
    two = ConstantExpression(2)
    expr = EqualExpression(one, two)
    with pytest.raises(ValueError):
        expr.evaluate()
    with pytest.raises(ValueError):
        expr.operate(one, two)


def test_expressions_binary_errors():
    child = BinaryExpression()
    with pytest.raises(NotImplementedError):
        child.name
    with pytest.raises(ValueError):
        child.evaluate()


def test_expressions_unary_specify_child_side():
    child = ConstantExpression(1337)
    expr = UnaryExpression(child, child_on_left=False)
    assert expr.get_child() == child
    assert expr.left is None
    assert expr.right == child


def test_expressions_unary_evaluate_errors():
    child = ConstantExpression(1337)
    expr = UnaryExpression(None)
    with pytest.raises(ValueError):
        expr.evaluate()


@pytest.mark.parametrize(
    "text", ["4/x^3+2-7x*12=0", "abs(-4) + abs(34)", "-sgn(-1) / sgn(2)", "sgn(0)"]
)
def test_expressions_to_math_ml(text: str):
    expr = ExpressionParser().parse(text)
    ml_string = expr.to_math_ml()
    assert "<math xmlns='http:#www.w3.org/1998/Math/MathML'>" in ml_string
    assert "</math>" in ml_string


def test_expressions_find_id():
    expr: MathExpression = ExpressionParser().parse("4 / x")
    node: MathExpression = expr.find_type(VariableExpression)[0]
    assert expr.find_id(node.id) == node


@pytest.mark.parametrize("visit_order", ["preorder", "inorder", "postorder"])
def test_expressions_to_list(visit_order: str):
    expr: MathExpression = ExpressionParser().parse("4 / x")
    assert len(expr.to_list(visit_order)) == 3


def test_expressions_to_list_errors():
    expr: MathExpression = ExpressionParser().parse("4 / x")
    with pytest.raises(ValueError):
        expr.to_list("invalid")


def test_expressions_clone():
    constant = ConstantExpression(4)
    assert constant.value == 4
    assert constant.clone().value == 4


def test_expressions_clone_root():
    a = ConstantExpression(1100)
    b = ConstantExpression(100)
    sub = SubtractExpression(a, b)
    c = ConstantExpression(300)
    add = AddExpression(sub, c)
    d = ConstantExpression(37)
    expr = AddExpression(add, d)
    assert expr.evaluate() == 1337


def test_expressions_function_exceptions():
    x = FunctionExpression()
    with pytest.raises(NotImplementedError):
        x.name


def test_expressions_variable_exceptions():
    x = VariableExpression(None)
    with pytest.raises(ValueError):
        str(x)
    with pytest.raises(ValueError):
        VariableExpression("x").evaluate({})
    with pytest.raises(ValueError):
        x.to_math_ml()

