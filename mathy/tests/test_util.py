from ..core.parser import ExpressionParser
from ..core.expressions import (
    ConstantExpression,
    VariableExpression,
    AddExpression,
    MultiplyExpression,
    DivideExpression,
    PowerExpression,
)
from ..core.util import is_preferred_term_form
from ..util import discount
from ..core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)


def test_is_preferred_term_form():
    examples = [
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


def test_reward_discounting():
    """Assert some things about the reward discounting to make sure we know how it impacts
    rewards for certain actions going back in time in episode history."""
    # ca cs dm -- ag -- | 15 | 0.6 | distributive factoring    | (((6 + 7) * y + 17x) + 17z) + 12y
    # -- cs -- -- ag -- | 14 | 0.4 | constant arithmetic       | ((13y + 17x) + 17z) + 12y
    # -- cs -- -- ag -- | 13 | 0.0 | commutative swap          | 12y + ((13y + 17x) + 17z)
    # -- cs -- -- ag -- | 12 | 0.0 | associative group         | (12y + (13y + 17x)) + 17z
    # -- cs -- df ag -- | 11 | 0.8 | associative group         | ((12y + 13y) + 17x) + 17z
    # ca cs dm -- ag -- | 10 | 0.2 | distributive factoring    | ((12 + 13) * y + 17x) + 17z
    # -- cs -- -- ag -- | 09 | 0.4 | constant arithmetic       | (25y + 17x) + 17z
    win_rewards = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 1.0]
    discounted_rewards = [0.8829603, 0.9019801, 0.92119205, 0.940598, 0.9602, 0.98, 1.0]
    rewards = discount(win_rewards)

    lose_rewards = [-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.02,-0.02,-0.02,-0.02,-0.02,-0.02,-1.0]
