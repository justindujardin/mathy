from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from ..core.expressions import MathTypeKeysMax
from ..state import MathyWindowObservation
from .swish_activation import swish


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        lstm_units: int,
        extract_window: Optional[int] = None,
        episode_reset_state_h: Optional[bool] = True,
        episode_reset_state_c: Optional[bool] = True,
        **kwargs,
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.extract_window = extract_window
        self.episode_reset_state_h = episode_reset_state_h
        self.episode_reset_state_c = episode_reset_state_c
        self.lstm_units = lstm_units
        self.init_rnn_state()
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.lstm_units,
            name="nodes_embedding",
            mask_zero=True,
        )
        self.in_transform = tf.keras.layers.Dense(
            self.units,
            name="in_transform",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.output_dense = tf.keras.layers.Dense(
            self.units,
            name="dense_embeddings_out",
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
            return_state=True,
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

    def call(
        self,
        features: MathyWindowObservation,
        burn_in_steps: int = 0,
        return_rnn_states: bool = False,
    ) -> Union[Tuple[tf.Tensor, int], Tuple[tf.Tensor, tf.Tensor]]:
        batch_size = len(features.nodes)  # noqa
        sequence_length = len(features.nodes[0])
        input = tf.convert_to_tensor(features.nodes)
        values = tf.convert_to_tensor(features.values)

        #
        # Contextualize nodes by expanding their integers to include (n) neighbors
        #
        if self.extract_window is not None:
            # reshape to 4 dimensions so we can use `extract_patches`
            input = tf.reshape(input, shape=[1, batch_size, sequence_length, 1])
            input_one = tf.image.extract_patches(
                images=input,
                sizes=[1, 1, self.extract_window, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )
            # Remove the extra dimensions
            input = tf.squeeze(input_one, axis=0)
        else:
            # Add an empty dimension (usually used by the extract window depth)
            input = tf.expand_dims(input, axis=-1)
            values = tf.expand_dims(values, axis=-1)

        query = self.token_embedding(input)
        query = tf.reshape(query, [batch_size, sequence_length, -1])
        query = tf.concat([query, values], axis=-1)
        query = self.in_transform(query)

        # Add context to each timesteps node vectors first
        in_state_h = tf.concat(features.rnn_state[0], axis=0)
        in_state_c = tf.concat(features.rnn_state[1], axis=0)
        query, nodes_state_h, nodes_state_c = self.nodes_lstm(
            query, initial_state=[in_state_h, in_state_c]
        )
        query = self.nodes_lstm_norm(query)

        state_h = tf.tile(nodes_state_h[-1:], [sequence_length, 1])
        state_c = tf.tile(nodes_state_c[-1:], [sequence_length, 1])
        query, state_h, state_c = self.time_lstm(
            query, initial_state=[state_h, state_c]
        )
        query = self.time_lstm_norm(query)

        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        # Concatenate the RNN output state with our historical RNN state
        # See: https://arxiv.org/pdf/1810.04437.pdf
        rnn_state_with_history = tf.concat(
            [state_h[-1:], features.rnn_history[0][0]], axis=-1
        )
        lstm_context = tf.tile(
            tf.expand_dims(rnn_state_with_history, axis=0),
            [batch_size, sequence_length, 1],
        )
        # Combine the output LSTM states with the historical average LSTM states
        # and concatenate it with the attended input sequence.
        sequence_out = tf.concat([query, lstm_context], axis=-1)
        sequence_out = self.output_dense(sequence_out)

        # For burn-in we only want the RNN states
        if return_rnn_states:
            return (state_h, state_c)

        # Return the embeddings and sequence length
        return (sequence_out, sequence_length)
