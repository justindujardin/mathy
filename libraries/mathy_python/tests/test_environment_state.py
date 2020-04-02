from mathy import MathyEnvState


def test_env_state():
    state = MathyEnvState(problem="4+4")
    assert state is not None


def test_env_state_encode_player():
    env_state = MathyEnvState(problem="4x+2")
    env_state = env_state.get_out_state(
        problem="2+4x", focus=0, moves_remaining=10, action=0
    )
    agent = env_state.agent
    assert agent.problem == "2+4x"
    assert agent.moves_remaining == 10
    assert agent.action == 0


def test_env_state_serialize_string():
    env_state = MathyEnvState(problem="4x+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="2+4x", focus=i, moves_remaining=10 - i, action=i
        )

    state_str = env_state.to_string()
    compare = MathyEnvState.from_string(state_str)
    assert env_state.agent.problem == compare.agent.problem
    assert env_state.agent.moves_remaining == compare.agent.moves_remaining
    for one, two in zip(env_state.agent.history, compare.agent.history):
        assert one.raw == two.raw
        assert one.focus == two.focus
        assert one.action == two.action


def test_env_state_serialize_numpy():
    env_state = MathyEnvState(problem="4x+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="2+4x", focus=i, moves_remaining=10 - i, action=i
        )

    state_np = env_state.to_np()
    compare = MathyEnvState.from_np(state_np)
    assert env_state.agent.problem == compare.agent.problem
    assert env_state.agent.moves_remaining == compare.agent.moves_remaining
    for one, two in zip(env_state.agent.history, compare.agent.history):
        assert one.raw == two.raw
        assert one.focus == two.focus
        assert one.action == two.action


def test_env_state_to_observation():
    """to_observation has defaults to allow calling with no arguments"""
    env_state = MathyEnvState(problem="4x+2")
    assert env_state.to_observation() is not None
