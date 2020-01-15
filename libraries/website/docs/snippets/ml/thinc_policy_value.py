import tensorflow as tf

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.policy_value_model import PolicyValueModel, get_or_create_policy_model
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(
    env.get_initial_state()[0], rnn_size=args.lstm_units
)
model = get_or_create_policy_model(args, predictions=env.action_size)
inputs = observations_to_window([observation]).to_inputs()
# predict_next only returns a policy for the last observation
# in the sequence, and applies masking and softmax to the output
policy, value, masked = model.predict([inputs])

# The policy is a 2D array of size (observations, num_nodes, actions)
assert policy.shape == (1, len(observation.nodes), env.action_size,)
# There should be one floating point output Value
assert isinstance(float(value), float)

# Save/Load the model
model = model.from_bytes(model.to_bytes())

# Forward pass
policy, value, masked = model.predict([inputs])

assert policy.shape == (1, len(observation.nodes), env.action_size,)
assert isinstance(float(value), float)
