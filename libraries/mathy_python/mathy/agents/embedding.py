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


class MathyEmbedding(tf.keras.Model):
    def __init__(self, config: BaseConfig, **kwargs):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.config = config
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.config.embedding_units,
            name="nodes_embedding",
            mask_zero=True,
        )

        # +1 for the value
        # +1 for the time
        # +2 for the problem type hashes
        self.concat_size = (
            4
            if self.config.use_env_features
            else 1
            if self.config.use_node_values
            else 0
        )
        if self.config.use_env_features:
            self.time_dense = tf.keras.layers.Dense(
                self.config.units, name="time_input"
            )
            self.type_dense = tf.keras.layers.Dense(
                self.config.units, name="type_input"
            )
        self.in_dense = tf.keras.layers.Dense(
            self.config.units,
            # In transform gets the embeddings concatenated with the
            # floating point value at each node.
            input_shape=(None, self.config.embedding_units + self.concat_size),
            name="in_dense",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.output_dense = tf.keras.layers.Dense(
            self.config.embedding_units,
            name="out_dense",
            activation="relu",
            kernel_initializer="he_normal",
        )

        NormalizeClass = tf.keras.layers.LayerNormalization
        if self.config.normalization_style == "batch":
            NormalizeClass = tf.keras.layers.BatchNormalization
        self.out_dense_norm = NormalizeClass(name="out_dense_norm")
        self.nodes_lstm_norm = NormalizeClass(name="lstm_nodes_norm")
        self.lstm_nodes = tf.keras.layers.LSTM(
            self.config.lstm_units,
            name="nodes_lstm",
            time_major=False,
            return_sequences=True,
            return_state=True,
        )
        self.lstm_attention = tf.keras.layers.Attention()

    def compute_output_shape(self, input_shapes: List[tf.TensorShape]) -> Any:
        nodes_shape: tf.TensorShape = input_shapes[0]
        return tf.TensorShape(
            (
                nodes_shape.dims[0].value,
                nodes_shape.dims[1].value,
                self.config.embedding_units,
            )
        )

    def call(self, features: MathyInputsType, train: tf.Tensor = None) -> tf.Tensor:
        nodes = features[ObservationFeatureIndices.nodes]
        values = features[ObservationFeatureIndices.values]
        type = features[ObservationFeatureIndices.type]
        time = features[ObservationFeatureIndices.time]
        nodes_shape = tf.shape(features[ObservationFeatureIndices.nodes])
        batch_size = nodes_shape[0]  # noqa
        sequence_length = nodes_shape[1]

        with tf.name_scope("prepare_inputs"):
            values = tf.expand_dims(values, axis=-1, name="values_input")
            query = self.token_embedding(nodes)
            # If not using env features, only concatenate the tokens and values
            if not self.config.use_env_features and self.config.use_node_values:
                query = tf.concat([query, values], axis=-1, name="tokens_and_values")
                query.set_shape(
                    (None, None, self.config.embedding_units + self.concat_size)
                )
            elif self.config.use_env_features:
                env_inputs = [
                    query,
                    self.type_dense(type),
                    self.time_dense(time),
                ]
                if self.config.use_node_values:
                    env_inputs.insert(1, values)

                query = tf.concat(env_inputs, axis=-1, name="tokens_and_values")

        # Input dense transforms
        query = self.in_dense(query)

        if self.config.use_lstm:
            output, state_h, state_c = self.lstm_nodes(query)
            output = self.lstm_attention([output, state_h])
            output = self.nodes_lstm_norm(output)
        else:
            # Non-recurrent model
            output = self.densenet(query)
            output = self.dense_attention([output, output])

        return self.out_dense_norm(output)
