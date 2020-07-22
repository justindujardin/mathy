import tensorflow as tf

from mathy import envs
from mathy.agent.config import AgentConfig
from mathy.agent.model import build_agent_model
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = AgentConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
model = build_agent_model(args, predictions=env.action_size)
inputs = observations_to_window([observation]).to_inputs()
policy, value, reward = model.predict(inputs)
# TODO: this is broken until the model is restructured to produce a single output

# The policy is a 1D array of size (actions * num_nodes)
assert policy.shape == (1, len(observation.nodes), env.action_size)

# There should be one floating point output Value
assert value.shape == (1, 1)
assert isinstance(float(value), float)
