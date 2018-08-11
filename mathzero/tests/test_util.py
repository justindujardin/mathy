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
        ("29y + (8 + 144y)", True),
        ("10z + (8 + 44z)", True),
        ("((1 + 9z) + 6) + 6z", True),
        ("x + 2(x – [3x – 8] + 3)", True),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        assert isPreferredTermForm(expr) == expected and input == input

