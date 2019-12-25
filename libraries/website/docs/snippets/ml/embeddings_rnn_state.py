import tensorflow as tf
import numpy as np

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.embedding import build_math_embeddings_model, EmbeddingsState
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(
    env.get_initial_state()[0], rnn_size=args.lstm_units
)

model: tf.keras.Model = build_math_embeddings_model(args, return_states=True)

inputs = observations_to_window([observation]).to_inputs()
# Predict over an observation sequence
embeddings, state_h, state_c = model.predict(inputs)

# Expect that the RNN states are non-zero
assert np.count_nonzero(state_h) > 0
assert np.count_nonzero(state_c) > 0
