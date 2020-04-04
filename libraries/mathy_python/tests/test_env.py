from mathy.state import MathyEnvState
from mathy.env import MathyEnv
from mathy.envs.poly_simplify import PolySimplify
from mathy.util import is_terminal_transition, EnvRewards
import random
import pytest


def test_mathy_env_init():
    env = MathyEnv()
    assert env is not None
    # Default env is abstract and cannot be directly used for problem solving
    with pytest.raises(NotImplementedError):
        env.get_initial_state()
    with pytest.raises(NotImplementedError):
        env.get_env_namespace()


def test_mathy_env_invalid_action_behaviors():

    problem = "4x + 2x"
    env = MathyEnv(error_invalid=True)
    env_state = MathyEnvState(problem=problem, max_moves=35)
    invalid_actions = [
        i for i, value in enumerate(env.get_valid_moves(env_state)) if value == 0
    ]
    random.shuffle(invalid_actions)
    action = invalid_actions[0]
    # error_invalid throws if an invalid action is selected
    with pytest.raises(ValueError):
        env.get_next_state(env_state, action)

    env = MathyEnv(error_invalid=False)
    env_state, transition, changed = env.get_next_state(env_state, action)
    # a transition is returned with error_invalid=False
    assert transition.reward == EnvRewards.INVALID_MOVE


def test_mathy_env_terminal_conditions():

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

    env = PolySimplify()
    for text, is_win in expectations + out_of_scope_valid:
        env_state = MathyEnvState(problem=text)
        reward = env.get_state_transition(env_state)
        assert text == text and env.is_terminal_state(env_state) == bool(is_win)
        assert text == text and is_terminal_transition(reward) == bool(is_win)


@pytest.mark.parametrize("pretty", [True, False])
def test_print_history(pretty: bool):
    env = PolySimplify()
    env_state = MathyEnvState(problem="4x+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="2+4x", focus=i, moves_remaining=10 - i, action=i
        )
    env.print_history(env_state, pretty=pretty)


def test_env_finalize_state():
    env = PolySimplify()

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="1337", action=1, focus=-1, moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="4x + 2", action=1, focus=-1, moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="4x + 2y", action=1, focus=-1, moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)
