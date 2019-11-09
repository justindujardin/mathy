from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from ..core.expressions import MathTypeKeysMax
from ..state import MathyWindowObservation
from .tensorflow.layers.multi_head_attention_stack import MultiHeadAttentionStack
from .tensorflow.layers.positional_embedding import TrigPosEmbedding
from .tensorflow.swish import swish


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        lstm_units: int,
        extract_window: Optional[int] = None,
        **kwargs
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
        self.attend_size = self.lstm_units // 2
        self.attn_bottleneck = tf.keras.layers.Dense(
            self.attend_size,
            name="in_bottleneck",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.bottleneck = tf.keras.layers.Dense(
            self.lstm_units,
            name="out_bottleneck",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.bottleneck_norm = tf.keras.layers.LayerNormalization(
            name="combined_features_normalize"
        )
        self.positional_embedding = TrigPosEmbedding(name="token_pos_embed")
        self.attention = MultiHeadAttentionStack(
            num_heads=4,
            num_layers=1,
            name="self_attention",
            attn_width=self.attend_size,
            return_attention=True,
        )
        self.time_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            name="lstm",
            return_sequences=True,
            time_major=True,
            return_state=True,
            unroll=True,
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
                rnn_history=burn_in_window.rnn_history,
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
        query = self.positional_embedding(query)

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
        # Repeat the time values for each node in the sequence
        #
        # [Batch, 1, 1] => [Batch, len(Sequence), 1]
        time_tiled = tf.tile(time_with_batch, [1, sequence_length, 1])

        # Combine the LSTM outputs with the contextual features
        time_out = tf.concat(
            [
                tf.cast(type_tiled, dtype=tf.float32),
                time_tiled,
                query,
                # tf.cast(move_mask + hint_mask, dtype=tf.float32),
            ],
            axis=-1,
        )

        # Transform our combined features into a consistent size
        time_out = self.attn_bottleneck(time_out)
        # Perform self-attention over the features
        time_out, self.attention_weights = self.attention(
            [time_out, time_out, time_out]
        )
        # Reduce the nodes sequence dimension to avoid having a large tiled
        # LSTM hidden/cell state.
        time_out = tf.expand_dims(tf.reduce_sum(time_out, axis=1), axis=1)

        state_h = features.rnn_state[0][0]
        state_c = features.rnn_state[1][0]
        time_out, state_h, state_c = self.time_lstm(
            time_out, initial_state=[state_h, state_c]
        )

        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        # Concatenate the RNN output state with our historical RNN state
        # See: https://arxiv.org/pdf/1810.04437.pdf
        rnn_state_with_history = tf.concat(
            [
                state_h,
                state_c,
                features.rnn_history[0][-1],
                features.rnn_history[1][-1],
            ],
            axis=1,
        )
        sequence_out = tf.tile(
            tf.expand_dims(rnn_state_with_history, axis=0),
            [batch_size, sequence_length, 1],
        )
        # we flattened the input state and tiled it, add positional information
        # back in before the final transformation.
        sequence_out = self.positional_embedding(sequence_out)
        sequence_out = self.bottleneck(sequence_out)

        # For burn-in we only want the RNN states
        if return_rnn_states:
            return (state_h, state_c)

        # Return the embeddings and sequence length
        return (sequence_out, sequence_length)
