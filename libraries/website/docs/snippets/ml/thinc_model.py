import os
import shutil
import tempfile
from typing import Tuple

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.embedding import MathyEmbedding
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window
from thinc.api import TensorFlowWrapper, keras_subclass
from thinc.layers import Linear, ReLu, Softmax, chain, with_list
from thinc.model import Model
from thinc.shims.tensorflow import TensorFlowShim
from thinc.types import Array, Array1d, Array2d, ArrayNd

args = BaseConfig()
env: MathyEnv = envs.PolySimplify()
observation: MathyObservation = env.state_to_observation(
    env.get_initial_state()[0], rnn_size=args.lstm_units
)

# output shape is: [num_observations, max_nodes_len, embedding_dimensions]
window = observations_to_window([observation, observation])
inputs = window.to_inputs()
input_shape = window.to_input_shapes()


@keras_subclass(
    "MathyEmbedding",
    X=window.to_inputs(),
    Y=window.mask,
    input_shape=input_shape,
    args={"config": args},
)
class ThincEmbeddings(MathyEmbedding):
    pass


embeddings = TensorFlowWrapper(ThincEmbeddings(args))

Y = embeddings.predict([inputs])

# serialize and restore the model
embeddings = embeddings.from_bytes(embeddings.to_bytes())

Y = embeddings.predict([inputs])
# We provided two observations in a sequence
assert Y.shape[0] == 2
# There are as many outputs as input sequences
assert Y.shape[1] == len(observation.nodes)
# Outputs vectors with the provided embedding units
assert Y.shape[-1] == args.embedding_units
