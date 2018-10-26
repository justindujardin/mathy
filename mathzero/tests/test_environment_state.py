from ..environment_state import EnvironmentAdapter, MathEnvironmentState
from ..core.parser import ExpressionParser


def test_math_state():
    state = EnvironmentAdapter()
    assert state is not None


def test_math_state_encode_player():
    state = EnvironmentAdapter()
    env_state = MathEnvironmentState(width=128, problem="4x+2")
    env_state = state.encode_player(env_state, "2+4x", 10)
    agent = env_state.agent
    assert agent.problem == "2+4x"
    assert agent.move_count == 10

