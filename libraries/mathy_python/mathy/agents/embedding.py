from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf

from mathy.agents.base_config import BaseConfig
from mathy.core.expressions import MathTypeKeysMax
from mathy.state import (
    MathyInputsType,
    MathyWindowObservation,
    ObservationFeatureIndices,
)
from mathy.agents.densenet import DenseNetStack

from .attention import SeqSelfAttention
from tf_siren import SinusodialRepresentationDense, SIRENModel


class MathyEmbedding(tf.keras.Model):
    def __init__(self, config: BaseConfig, **kwargs):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.config = config
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.config.embedding_units,
            name="nodes_input",
            mask_zero=True,
        )
        # +1 for the value
        # +2 for the problem type hashes
        self.values_dense = SinusodialRepresentationDense(
            self.config.units, name="values_input"
        )
        self.type_dense = SinusodialRepresentationDense(
            self.config.units, name="type_input"
        )
        self.time_dense = SinusodialRepresentationDense(
            self.config.units, name="time_input"
        )
        self.siren_mlp = SIRENModel(
            units=self.config.units,
            final_units=self.config.units,
            num_layers=2,
            # In transform gets the embeddings concatenated with the
            # floating point value at each node.
            # input_shape=(None, self.config.embedding_units + self.concat_size),
            name="siren",
        )

    def call(self, features: MathyInputsType, train: tf.Tensor = None) -> tf.Tensor:
        output = tf.concat(
            [
                self.token_embedding(features[ObservationFeatureIndices.nodes]),
                self.values_dense(
                    tf.expand_dims(features[ObservationFeatureIndices.values], axis=-1)
                ),
                self.type_dense(features[ObservationFeatureIndices.type]),
                self.time_dense(features[ObservationFeatureIndices.time]),
            ],
            axis=-1,
            name="input_vectors",
        )
        output = self.siren_mlp(output)
        return output, tf.zeros((10, 10))
