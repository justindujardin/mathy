import numpy as np

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.embedding import MathyEmbedding
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window

args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(
    env.get_initial_state()[0], rnn_size=args.lstm_units
)
model = MathyEmbedding(args)
inputs = observations_to_window([observation]).to_inputs()

# Expect that the RNN states are zero to begin
assert np.count_nonzero(model.state_h.numpy()) == 0
assert np.count_nonzero(model.state_c.numpy()) == 0

embeddings = model.call(inputs)

# Expect that the RNN states are non-zero
assert np.count_nonzero(model.state_h.numpy()) > 0
assert np.count_nonzero(model.state_c.numpy()) > 0

# You can reset them
model.reset_rnn_state()

# Expect that the RNN states are zero again
assert np.count_nonzero(model.state_h.numpy()) == 0
assert np.count_nonzero(model.state_c.numpy()) == 0
