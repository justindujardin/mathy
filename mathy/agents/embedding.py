from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from .tensorflow.layers.multi_head_attention_stack import MultiHeadAttentionStack
from ..core.expressions import MathTypeKeysMax
from ..state import MathyWindowObservation


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, units: int, lstm_units: int, extract_window: Optional[int] = 3, **kwargs
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.extract_window = extract_window
        self.lstm_units = lstm_units
        self.init_rnn_state()
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.lstm_units,
            name="nodes_embedding",
            mask_zero=True,
        )
        self.bottleneck = tf.keras.layers.Dense(
            self.lstm_units, name="combined_features", activation="relu"
        )
        self.bottleneck_norm = tf.keras.layers.BatchNormalization(
            name="combined_features_normalize"
        )
        self.attention = MultiHeadAttentionStack(
            num_heads=8, num_layers=3, name="self_attention", attn_width=self.lstm_units
        )
        self.time_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            name="timestep_lstm",
            return_sequences=True,
            time_major=True,
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

    def reset_rnn_state(self):
        """Zero out the RNN state for a new episode"""
        self.state_c.assign(tf.zeros([1, self.lstm_units]))
        self.state_h.assign(tf.zeros([1, self.lstm_units]))

    def call(
        self, features: MathyWindowObservation, burn_in_steps=0, return_rnn_states=False
    ) -> Union[Tuple[tf.Tensor, int], Tuple[tf.Tensor, tf.Tensor]]:
        batch_size = len(features.nodes)  # noqa
        sequence_length = len(features.nodes[0])
        type = tf.convert_to_tensor(features.type)
        time = tf.convert_to_tensor(features.time)
        input = tf.convert_to_tensor(features.nodes)

        # Do any specified burn-in steps (for off-policy RNN state correction)
        burn_in_window = features
        burn_in_state_h = features.rnn_state[0]
        burn_in_state_c = features.rnn_state[1]
        for i in range(burn_in_steps):
            burn_in_window = MathyWindowObservation(
                nodes=burn_in_window.nodes,
                mask=burn_in_window.mask,
                hints=burn_in_window.hints,
                type=burn_in_window.type,
                time=burn_in_window.time,
                rnn_state=[burn_in_state_h, burn_in_state_c],
            )
            result: Tuple[tf.Tensor, tf.Tensor] = self.call(
                burn_in_window, burn_in_steps=0, return_rnn_states=True
            )
            burn_in_state_h, burn_in_state_c = result

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

        query = self.token_embedding(input)
        query = tf.reshape(query, [batch_size, sequence_length, -1])

        #
        # Aux features
        #
        move_mask = tf.convert_to_tensor(features.mask)
        hint_mask = tf.convert_to_tensor(features.hints)
        if batch_size == 1:
            move_mask = move_mask[tf.newaxis, :, :]
            hint_mask = hint_mask[tf.newaxis, :, :]
        move_mask = tf.reshape(move_mask, [batch_size, sequence_length, -1])
        hint_mask = tf.reshape(hint_mask, [batch_size, sequence_length, -1])

        # Reshape the "type" information and combine it with each node in the
        # sequence so the nodes have context for the current task
        #
        # [Batch, len(Type)] => [Batch, 1, len(Type)]
        type_with_batch = type[:, tf.newaxis, :]
        # Repeat the type values for each node in the sequence
        #
        # [Batch, 1, len(Type)] => [Batch, len(Sequence), len(Type)]
        type_tiled = tf.tile(type_with_batch, [1, sequence_length, 1])

        # Reshape the "time" information so it has a time axis
        #
        # [Batch, 1] => [Batch, 1, 1]
        time_with_batch = time[:, tf.newaxis, :]
        # Repeat the type values for each node in the sequence
        #
        # [Batch, 1, 1] => [Batch, len(Sequence), 1]
        time_tiled = tf.tile(time_with_batch, [1, sequence_length, 1])

        # Combine the LSTM outputs with the contextual features
        time_out = tf.concat(
            [
                tf.cast(type_tiled, dtype=tf.float32),
                time_tiled,
                query,
                tf.cast(move_mask + hint_mask, dtype=tf.float32),
            ],
            axis=-1,
        )
        # use a bottleneck so that we know the dimensions fit the attention layer below
        time_out = self.bottleneck_norm(self.bottleneck(time_out))

        # Self-attention
        time_out = self.attention([time_out, time_out, time_out])

        # LSTM for temporal dependencies
        state_h = tf.tile(features.rnn_state[0][-1], [sequence_length, 1])
        state_c = tf.tile(features.rnn_state[1][-1], [sequence_length, 1])
        time_out, state_h, state_c = self.time_lstm(
            time_out, initial_state=[state_h, state_c]
        )
        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        # For burn-in we only want the RNN states
        if return_rnn_states:
            state_h = tf.tile(state_h[-1:], [batch_size, 1])
            state_c = tf.tile(state_c[-1:], [batch_size, 1])
            return (state_h, state_c)

        # Return the embeddings and sequence length
        return (time_out, sequence_length)
