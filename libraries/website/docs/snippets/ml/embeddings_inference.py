import tensorflow as tf

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.embedding import build_math_embeddings_model
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(
    env.get_initial_state()[0], rnn_size=args.lstm_units
)
model: tf.keras.Model = build_math_embeddings_model(args)
# output shape is: [num_observations, max_nodes_len, embedding_dimensions]
embeddings = model.predict(observations_to_window([observation]).to_inputs())

# We only passed one observation sequence
assert embeddings.shape[0] == 1
# There are as many outputs as input sequences
assert embeddings.shape[1] == len(observation.nodes)
# Outputs vectors with the provided embedding units
assert embeddings.shape[-1] == args.embedding_units
