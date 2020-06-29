from mathy import envs
from mathy.agents.config import AgentConfig
from mathy.agents.embedding import MathyEmbedding
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window


args = AgentConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
model = MathyEmbedding(args)
# output shape is: [num_observations, max_nodes_len, embedding_dimensions]
inputs = observations_to_window([observation, observation]).to_inputs()
embeddings, attentions = model(inputs)

# We provided two observations in a sequence
assert embeddings.shape[0] == 2
# There are as many outputs as input sequences
assert embeddings.shape[1] == len(observation.nodes)
# Outputs vectors with the provided embedding units
assert embeddings.shape[-1] == args.embedding_units

# The attention output is a grid of [num_observations, max_nodes_len, max_nodes_len]
assert attentions.shape == [2, len(observation.nodes), len(observation.nodes)]

