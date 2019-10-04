from ..state import MathyEnvState
from ..mathy_env import MathyEnv
from ..envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from ..util import is_terminal_transition
import random
import pytest


def test_mathy_env_init():
    env = MathyEnv()
    assert env is not None
    # Default env is abstract and cannot be directly used for problem solving
    with pytest.raises(NotImplementedError):
        env.get_initial_state()


def test_mathy_env_jd():
    env = MathyEnv()
    assert env is not None
    problem = "5y * 9x + 8z + 8x + 3z * 10y * 11x + 10y"
    env_state = MathyEnvState(problem=problem, max_moves=35)
    for i in range(3):
        actions = env.get_valid_moves(env_state)
        indices = [i for i, value in enumerate(actions) if value == 1]
        random.shuffle(indices)
        env_state, value, changed = env.get_next_state(env_state, indices[0])
    assert env_state.to_input_features([]) is not None


def test_binomial_distribution_env():

    expectations = [
        ("70656 * (x^2 * z^6)", True),
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("4x^2", True),
        ("100y * x + 2", True),
        ("10y * 10x + 2", False),
        ("10y + 1000y * (y * z)", False),
        ("4 * (5y + 2)", False),
        ("2", True),
        ("4x * 2", False),
        ("4x * 2x", False),
        ("4x + 2x", False),
        ("4 + 2", False),
        ("3x + 2y + 7", True),
        ("3x^2 + 2x + 7", True),
        ("3x^2 + 2x^2 + 7", False),
    ]

    # Valid solutions but out of scope so they aren't counted as wins.
    #
    # This works because the problem sets exclude this type of > 2 term
    # polynomial expressions
    out_of_scope_valid = []

    env = MathyPolynomialSimplificationEnv()
    for text, is_win in expectations + out_of_scope_valid:
        env_state = MathyEnvState(problem=text)
        reward = env.get_state_transition(env_state)
        assert text == text and is_terminal_transition(reward) == bool(is_win)
