from ..core.parser import ExpressionParser
from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    MultiplyExpression,
    DivideExpression,
    PowerExpression,
)
from ..rules.helpers import (
    is_preferred_term_form,
    has_like_terms,
    TermEx,
    get_term_ex,
    get_sub_terms,
)
from ..util import discount
from ..rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)


def test_get_sub_terms():
    expectations = [
        ("70656 * (x^2 * z^6)", 2),
        ("4x^2 * z^6 * y", 3),
        ("2x^2", 1),
        ("x^2", 1),
        ("2", 1),
    ]
    invalid_expectations = [
        # can't have more than one term
        ("4 + 4", False)
    ]
    parser = ExpressionParser()
    for text, output in expectations + invalid_expectations:
        exp = parser.parse(text)
        sub_terms = get_sub_terms(exp)
        if output is False:
            assert text == text and sub_terms == output
        else:
            assert text == text and len(sub_terms) == output


def test_get_term_ex():
    examples = [
        ("4x^2", TermEx(4, "x", 2)),
        ("4x", TermEx(4, "x", None)),
        ("x", TermEx(None, "x", None)),
        # TODO: non-natural term forms? If this is supported we can drop the other
        #       get_term impl maybe?
        # ("x * 2", TermEx(2, "x", None)),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        expr = parser.parse(input)
        assert input == input and get_term_ex(expr) == expected


def test_is_preferred_term_form():
    examples = [
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("4x * z", True),
        ("z * 4x", True),
        ("2x * x", False),
        ("29y", True),
        ("z", True),
        ("z * 10", False),
        ("4x^2", True),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        expr = parser.parse(input)
        assert input == input and is_preferred_term_form(expr) == expected


def test_has_like_terms():
    examples = [
        ("14 + 6y + 7x + x * (3y)", False),
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("100y * x + 2", False),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        expr = parser.parse(input)
        assert input == input and has_like_terms(expr) == expected
