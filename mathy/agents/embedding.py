from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
import json
from ..core.expressions import MathTypeKeysMax
from ..state import MathyWindowObservation, MathyInputs, MathyInputsType
from .swish_activation import swish


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        lstm_units: int,
        extract_window: Optional[int] = None,
        episode_reset_state_h: Optional[bool] = False,
        episode_reset_state_c: Optional[bool] = False,
        **kwargs,
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.extract_window = extract_window
        self.episode_reset_state_h = episode_reset_state_h
        self.episode_reset_state_c = episode_reset_state_c
        self.lstm_units = lstm_units
        self.init_rnn_state()
        self.embedding_units = self.lstm_units
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.embedding_units,
            name="nodes_embedding",
            mask_zero=True,
        )
        self.in_dense_expand = tf.keras.layers.Dense(
            self.units,
            # In transform gets the embeddings concatenated with the
            # floating point value at each node.
            input_shape=(None, self.embedding_units + 1),
            name="in_dense_expand",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.in_dense_compress = tf.keras.layers.Dense(
            self.units // 2,
            name="in_dense_compress",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.output_dense_expand = tf.keras.layers.Dense(
            self.units * 2,
            name="out_dense_expand",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.output_dense_compress = tf.keras.layers.Dense(
            self.units,
            name="out_dense_expand",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.time_lstm_norm = tf.keras.layers.LayerNormalization(name="lstm_time_norm")
        self.nodes_lstm_norm = tf.keras.layers.LayerNormalization(
            name="lstm_nodes_norm"
        )
        self.time_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            name="timestep_lstm",
            return_sequences=True,
            time_major=True,
            return_state=True,
        )
        self.nodes_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            name="nodes_lstm",
            return_sequences=True,
            time_major=False,
            return_state=False,
        )

    def init_rnn_state(self):
        """Track RNN states with variables in the graph"""
        self.state_c = tf.Variable(
            tf.zeros([1, self.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_c",
        )
        self.state_h = tf.Variable(
            tf.zeros([1, self.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_h",
        )

    def reset_rnn_state(self, force: bool = False):
        """Zero out the RNN state for a new episode"""
        if self.episode_reset_state_h or force is True:
            self.state_h.assign(tf.zeros([1, self.lstm_units]))
        if self.episode_reset_state_c or force is True:
            self.state_c.assign(tf.zeros([1, self.lstm_units]))

    def call(self, features: MathyInputsType) -> tf.Tensor:
        nodes = tf.convert_to_tensor(features[MathyInputs.nodes])
        values = tf.convert_to_tensor(features[MathyInputs.values])
        nodes_shape = tf.shape(nodes)
        batch_size = nodes_shape[0]  # noqa
        sequence_length = nodes_shape[1]
        in_rnn_state = features[MathyInputs.rnn_state][0]
        in_rnn_history = features[MathyInputs.rnn_history][0]

        with tf.name_scope("prepare_inputs"):
            #
            # Contextualize nodes by expanding their integers to include (n) neighbors
            #
            if self.extract_window is not None:
                # reshape to 4 dimensions so we can use `extract_patches`
                input = tf.reshape(
                    nodes,
                    shape=[1, batch_size, sequence_length, 1],
                    name="extract_window_in",
                )
                input_one = tf.image.extract_patches(
                    images=input,
                    sizes=[1, 1, self.extract_window, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                    name="extract_window_op",
                )
                # Remove the extra dimensions
                input = tf.squeeze(input_one, axis=0, name="extract_window_out")
            else:
                # Add an empty dimension (usually used by the extract window depth)
                input = tf.expand_dims(nodes, axis=-1, name="nodes_input")
                values = tf.expand_dims(values, axis=-1, name="values_input")

            query = self.token_embedding(input)
            query = tf.reshape(
                query, [batch_size, sequence_length, -1], name="tokens_batch_reshape"
            )
            query = tf.concat([query, values], axis=-1, name="tokens_and_values")
            query.set_shape((None, None, self.embedding_units + 1))

        # Input dense transforms
        query = self.in_dense_expand(query)
        query = self.in_dense_compress(query)

        with tf.name_scope("prepare_initial_states"):

            nodes_initial_h = tf.squeeze(
                tf.concat(in_rnn_state[0], axis=0, name="nodes_hidden"), axis=1
            )
            nodes_initial_c = tf.squeeze(
                tf.concat(in_rnn_state[1], axis=0, name="nodes_cell"), axis=1
            )
            time_initial_h = tf.tile(
                in_rnn_state[0][0], [sequence_length, 1], name="time_hidden"
            )
            time_initial_c = tf.tile(
                in_rnn_state[1][0], [sequence_length, 1], name="time_cell"
            )

        with tf.name_scope("rnn"):

            query = self.nodes_lstm(
                query, initial_state=[nodes_initial_h, nodes_initial_c]
            )
            query = self.nodes_lstm_norm(query)
            query, state_h, state_c = self.time_lstm(
                query, initial_state=[time_initial_h, time_initial_c]
            )
            query = self.time_lstm_norm(query)
            historical_state_h = tf.squeeze(
                tf.concat(in_rnn_history[0], axis=0, name="average_history_hidden"),
                axis=1,
            )

        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        # Concatenate the RNN output state with our historical RNN state
        # See: https://arxiv.org/pdf/1810.04437.pdf
        with tf.name_scope("combine_outputs"):
            rnn_state_with_history = tf.concat(
                [state_h[-1:], historical_state_h[-1:]], axis=-1,
            )
            lstm_context = tf.tile(
                tf.expand_dims(rnn_state_with_history, axis=0),
                [batch_size, sequence_length, 1],
            )
            # Combine the output LSTM states with the historical average LSTM states
            # and concatenate it with the query
            output = tf.concat([query, lstm_context], axis=-1, name="combined_outputs")

        # Sequence outputs
        output = self.output_dense_expand(output)
        output = self.output_dense_compress(output)

        # Return the embeddings
        return output

