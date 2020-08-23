from mathy import envs
from mathy.agent.config import AgentConfig
from mathy.agent.model import build_agent_model
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window


def test_model_call():

    args = AgentConfig()
    env: MathyEnv = envs.PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    model = build_agent_model(args, predictions=env.action_size)
    inputs = observations_to_window([observation], args.max_len).to_inputs()
    fn_policy, args_policy, value, reward = model.predict(inputs)
    # The function policy is a 2D array of size (None, len(env.rules))
    assert fn_policy.shape == (1, env.action_size)
    # The args policy determines which node in the sequence to apply the
    # function to. It's shape is (None, len(env.rules), max_seq_len)
    assert args_policy.shape == (1, env.action_size, len(observation.nodes))

    # The estimated Value is a float
    assert value.shape == (1, 1)
    assert isinstance(float(value), float)

    # The estimated reward is also a float
    assert reward.shape == (1, 1)
    assert isinstance(float(reward), float)
