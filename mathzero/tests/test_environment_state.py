from ..environment_state import EnvironmentAdapter, MathEnvironmentState
from ..core.parser import ExpressionParser


def test_math_state():
    state = EnvironmentAdapter()
    assert state is not None


def test_math_state_encode_player():
    state = EnvironmentAdapter()
    env_state = MathEnvironmentState(width=128, problem="4x+2")
    env_state = state.encode_player(env_state, 1, "2+4x", 10)
    agent = env_state.get_player(1)
    assert agent.problem == "2+4x"
    assert agent.move_count == 10


def test_math_state_get_canonical_board():
    env = EnvironmentAdapter()
    env_state = MathEnvironmentState(width=128, problem="4x+2")
    env_state = env.encode_player(env_state, -1, "2+4x", 1)
    env_state = env.get_canonical_board(env_state, -1)
    # The canonical env_state always represents the env_state from the same perspective, in this case
    # from the perspective of player 1. So player -1's canonical env_state will return the player -1
    # state when you decode player 1 from it.
    assert env_state.agent_one.move_count == 1
    assert env_state.agent_one.player == -1
