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
    def __init__(
        self,
        config: BaseConfig,
        episode_reset_state_h: Optional[bool] = True,
        episode_reset_state_c: Optional[bool] = True,
        **kwargs,
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.config = config
        self.episode_reset_state_h = episode_reset_state_h
        self.episode_reset_state_c = episode_reset_state_c
        self.init_rnn_state()
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
        # if self.config.use_node_values:
        #     self.values_dense = tf.keras.layers.Dense(self.config.units, name="values_input")
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
        if self.config.use_lstm:
            self.nodes_lstm_norm = NormalizeClass(name="lstm_nodes_norm")
            self.lstm = tf.keras.layers.LSTM(
                self.config.lstm_units,
                name="lstm",
                return_sequences=True,
                time_major=False,
                return_state=True,
            )
        else:
            self.densenet = DenseNetStack(
                units=self.config.units,
                num_layers=6,
                output_transform=self.output_dense,
                normalization_style=self.config.normalization_style,
            )
            self.dense_attention = tf.keras.layers.Attention()

    def compute_output_shape(self, input_shapes: List[tf.TensorShape]) -> Any:
        nodes_shape: tf.TensorShape = input_shapes[0]
        return tf.TensorShape(
            (
                nodes_shape.dims[0].value,
                nodes_shape.dims[1].value,
                self.config.embedding_units,
            )
        )

    def init_rnn_state(self):
        """Track RNN states with variables in the graph"""
        self.state_c = tf.Variable(
            tf.zeros([1, self.config.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_c",
        )
        self.state_h = tf.Variable(
            tf.zeros([1, self.config.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_h",
        )

    def reset_rnn_state(self, force: bool = False):
        """Zero out the RNN state for a new episode"""
        if self.episode_reset_state_h or force is True:
            self.state_h.assign(tf.zeros([1, self.config.lstm_units]))

        if self.episode_reset_state_c or force is True:
            self.state_c.assign(tf.zeros([1, self.config.lstm_units]))

    def call(self, features: MathyInputsType, train: tf.Tensor = None) -> tf.Tensor:
        nodes = features[ObservationFeatureIndices.nodes]
        values = features[ObservationFeatureIndices.values]
        type = features[ObservationFeatureIndices.type]
        time = features[ObservationFeatureIndices.time]
        nodes_shape = tf.shape(features[ObservationFeatureIndices.nodes])
        batch_size = nodes_shape[0]  # noqa
        sequence_length = nodes_shape[1]

        in_rnn_state_h = features[ObservationFeatureIndices.rnn_state_h]
        in_rnn_state_c = features[ObservationFeatureIndices.rnn_state_c]
        in_rnn_history_h = features[ObservationFeatureIndices.rnn_history_h]

        with tf.name_scope("prepare_inputs"):
            values = tf.expand_dims(values, axis=-1, name="values_input")
            query = self.token_embedding(nodes)

            # if self.config.use_node_values:
            #     values = self.values_dense(values)

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

                query = tf.concat(env_inputs, axis=-1, name="tokens_and_values",)

        # Input dense transforms
        query = self.in_dense(query)

        if self.config.use_lstm:
            with tf.name_scope("rnn"):
                query, state_h, state_c = self.lstm(
                    query, initial_state=[in_rnn_state_h, in_rnn_state_c]
                )
                query = self.nodes_lstm_norm(query)

            self.state_h.assign(state_h[-1:])
            self.state_c.assign(state_c[-1:])

            # Concatenate the RNN output state with our historical RNN state
            # See: https://arxiv.org/pdf/1810.04437.pdf
            with tf.name_scope("combine_outputs"):
                rnn_state_with_history = tf.concat(
                    [state_h[-1:], in_rnn_history_h[-1:]], axis=-1,
                )
                lstm_context = tf.tile(
                    tf.expand_dims(rnn_state_with_history, axis=0),
                    [batch_size, sequence_length, 1],
                )
                # Combine the output LSTM states with the historical average LSTM states
                # and concatenate it with the query
                output = tf.concat(
                    [query, lstm_context], axis=-1, name="combined_outputs"
                )
                output = self.output_dense(output)
        else:
            # Non-recurrent model
            output = self.densenet(query)
            output = self.dense_attention([output, output])

        return self.out_dense_norm(output)
