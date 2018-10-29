from ..core.parser import ExpressionParser
from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    MultiplyExpression,
    DivideExpression,
    PowerExpression,
)
from ..core.util import isPreferredTermForm
from ..core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)


def test_is_preferred_term_form():
    examples = [
        ("4xz", True),
        ("z * 4x", False),
        ("2x * x", False),
        ("29y", True),
        ("z", True),
        ("z * 10", False),
        ("4x^2", True),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        expr = parser.parse(input)
        assert input == input and isPreferredTermForm(expr) == expected
