from ..core.parser import ExpressionParser
from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    MultiplyExpression,
    DivideExpression,
    PowerExpression,
)
from ..core.util import is_preferred_term_form, has_like_terms, TermEx, get_term_ex
from ..util import discount
from ..core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)


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
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("100y * x + 2", False),
    ]
    parser = ExpressionParser()
    for input, expected in examples:
        expr = parser.parse(input)
        assert input == input and has_like_terms(expr) == expected


def test_reward_discounting():
    """Assert some things about the reward discounting to make sure we know how it impacts
    rewards for certain actions going back in time in episode history."""
    win_rewards = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 1.0]
    discounted_rewards = [0.8829603, 0.9019801, 0.92119205, 0.940598, 0.9602, 0.98, 1.0]
    rewards = discount(win_rewards)

    lose_rewards = [
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.02,
        -0.02,
        -0.02,
        -0.02,
        -0.02,
        -0.02,
        -1.0,
    ]
