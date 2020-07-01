from mathy import envs
from mathy.agents.config import AgentConfig
from mathy.agents.model import AgentModel
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = AgentConfig(use_env_features=True)
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
model = AgentModel(args, predictions=env.action_size)
inputs = observations_to_window([observation]).to_inputs()
# predict_next only returns a policy for the last observation
# in the sequence, and applies masking and softmax to the output
policy, value = model.predict_next(inputs)

# The policy is a 1D array of size (actions * num_nodes)
assert policy.shape.rank == 1
assert policy.shape == (env.action_size * len(observation.nodes),)

# There should be one floating point output Value
assert value.shape.rank == 0
assert isinstance(float(value.numpy()), float)
