import os
import shutil
import tempfile
from typing import List, Tuple

import numpy as np

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.embedding import MathyEmbedding
from mathy.env import MathyEnv
from mathy.state import MathyEnvState, MathyObservation, observations_to_window
from thinc.api import TensorFlowWrapper
from thinc.layers import (
    Embed,
    Linear,
    MeanPool,
    ReLu,
    Softmax,
    chain,
    list2ragged,
    with_array,
    with_list,
)
from thinc.model import Model
from thinc.shims.tensorflow import TensorFlowShim
from thinc.types import Array, Array1d, Array2d, ArrayNd

# Mathy env setup and initial observations
args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]
observation: MathyObservation = env.state_to_observation(
    state, rnn_size=args.lstm_units
)
window = observations_to_window([observation, observation])
inputs = window.to_inputs()

X = [inputs]  # TODO: why do I need to wrap inputs for the tf wrapper?
input_shape = window.to_input_shapes()
embeddings = TensorFlowWrapper(MathyEmbedding(args), input_shape=input_shape)
embeddings.initialize([inputs])

embed_Y = embeddings.predict([inputs])
# Shape = (2, 23, 128) = (num_observations, padded_sequence_len, vector_width)

# The policy head is a softmax(actions) for each node in the sequence.
policy_head = chain(embeddings, Softmax(6))
policy_head.initialize([inputs])
policy_Y = policy_head.predict([inputs])
# Shape (desired) = (2, 23, 6)

# The value head is normally a linear transformation from the
# output embedding layer's RNN state. I haven't tried mixing
# that tensor in here. TODO: try that
value_head = chain(embeddings, MeanPool(), Linear(1))
value_head.initialize([inputs])
value_Y = value_head.predict([inputs])
# Shape (desired) = (2, 1)

# Combined [policy_head, value_head] outputs without invoking embeddings twice?
model: Model[ArrayNd, Tuple[Array2d, Array1d]] = ...

model.initialize([inputs])

Y = model.predict([inputs])
model.to_disk(f"training/model")
