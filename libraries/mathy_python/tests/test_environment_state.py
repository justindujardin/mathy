from ..mathy import MathyEnvState


def test_math_state():
    state = MathyEnvState(problem="4+4")
    assert state is not None


def test_math_state_encode_player():
    env_state = MathyEnvState(problem="4x+2")
    env_state = env_state.get_out_state(
        problem="2+4x", focus_index=3, moves_remaining=10, action=0
    )
    agent = env_state.agent
    assert agent.problem == "2+4x"
    assert agent.moves_remaining == 10
    assert agent.action == 0
    assert agent.focus_index == 3


def test_math_state_to_observation():
    """to_observation has defaults to allow calling with no arguments"""
    env_state = MathyEnvState(problem="4x+2")
    assert env_state.to_observation() is not None
