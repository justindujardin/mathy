from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from ..agent.layers.keras_self_attention.multi_head_attention import MultiHeadAttention
from ..agent.layers.resnet_stack import ResNetStack
from ..core.expressions import MathTypeKeysMax
from ..features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
)
from ..state import (
    PROBLEM_TYPE_HASH_BUCKETS,
    MathyBatchObservation,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 128,
        lstm_units: int = 256,
        burn_in_steps: int = 10,
        extract_window: Optional[int] = None,
        encode_tokens_with_type: bool = False,
        **kwargs,
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.encode_tokens_with_type = encode_tokens_with_type
        self.extract_window = extract_window
        self.lstm_units = lstm_units
        self.burn_in_steps = burn_in_steps
        self.init_rnn_state()
        token_in_dimension = (
            PROBLEM_TYPE_HASH_BUCKETS
            if self.encode_tokens_with_type
            else MathTypeKeysMax
        )
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=token_in_dimension,
            output_dim=self.lstm_units,
            name="nodes_embedding",
            mask_zero=True,
        )
        self.mask_embedding = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=self.lstm_units,
            name="mask_embedding",
            mask_zero=False,
        )
        self.hints_embedding = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=self.lstm_units,
            name="hints_embedding",
            mask_zero=False,
        )
        self.attention = MultiHeadAttention(head_num=8, name="multi_head_attention")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.concat = tf.keras.layers.Concatenate(name=f"embedding_concat")
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
        self.embedding = tf.keras.layers.Dense(units=self.units, name="embedding")

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
        self, features: MathyWindowObservation, do_burn_in=False
    ) -> Tuple[tf.Tensor, int]:
        batch_size = len(features.nodes)  # noqa
        sequence_length = len(features.nodes[0])
        type = tf.convert_to_tensor(features.type)
        input = tf.convert_to_tensor(features.nodes)

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

        if self.encode_tokens_with_type:
            # Reshape the "type" information and combine it with each node in the
            # sequence so the nodes have context for the current task
            #
            # [Batch, len(Type)] => [Batch, 1, len(Type)]
            type_with_batch = type[:, tf.newaxis, :]
            # Repeat the type values for each node in the sequence
            #
            # [Batch, 1, len(Type)] => [Batch, len(Sequence), len(Type)]
            type_tiled = tf.tile(type_with_batch, [1, sequence_length, 1])
            # Concatenate them yielding nodes with length [seq + extract_len + type_len]
            input = tf.concat([type_tiled, input], axis=2)

        query = self.token_embedding(input)
        query = tf.reshape(query, [batch_size, sequence_length, -1])

        # Embed the valid move mask for the attention layer
        value = self.mask_embedding(tf.convert_to_tensor(features.mask))
        key = self.hints_embedding(tf.convert_to_tensor(features.hints))

        # Add context to each timesteps node vectors first
        in_state_h = tf.concat(features.rnn_state[0], axis=0)
        in_state_c = tf.concat(features.rnn_state[1], axis=0)
        query, state_h, state_c = self.nodes_lstm(
            query, initial_state=[in_state_h, in_state_c]
        )

        state_h = tf.tile(state_h[-1:], [sequence_length, 1])
        state_c = tf.tile(state_c[-1:], [sequence_length, 1])
        time_out, state_h, state_c = self.time_lstm(
            query, initial_state=[state_h, state_c]
        )
        time_out = self.attention([time_out, value, key])

        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        return (time_out, sequence_length)
