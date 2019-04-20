from ..environment_state import MathEnvironmentState
from ..core.parser import ExpressionParser


def test_math_state():
    state = MathEnvironmentState(problem="4+4")
    assert state is not None


def test_math_state_encode_player():
    env_state = MathEnvironmentState(problem="4x+2")
    env_state = env_state.encode_player(
        problem="2+4x", focus_index=3, moves_remaining=10, action=0
    )
    agent = env_state.agent
    assert agent.problem == "2+4x"
    assert agent.moves_remaining == 10
    assert agent.action == 0
    assert agent.focus_index == 3

