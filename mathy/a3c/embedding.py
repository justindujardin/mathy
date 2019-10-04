from typing import Any, Dict, Tuple

import tensorflow as tf

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
    MathyBatchObservation,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
    PROBLEM_TYPE_HASH_BUCKETS,
)


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, units: int = 128, lstm_units: int = 256, burn_in_steps: int = 10, **kwargs
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.lstm_units = lstm_units
        self.burn_in_steps = burn_in_steps
        self.init_rnn_state()
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=PROBLEM_TYPE_HASH_BUCKETS,
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
        self.attention = tf.keras.layers.AdditiveAttention(name="attention")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.concat = tf.keras.layers.Concatenate(name=f"embedding_concat")
        self.input_dense = tf.keras.layers.Dense(
            self.units, name="embedding_input", use_bias=False
        )
        self.resnet = ResNetStack(units=units, name="embedding_resnet", num_layers=3)
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
        self.nodes_lstm_bwd = tf.keras.layers.LSTM(
            self.lstm_units,
            name="nodes_lstm_bwd",
            return_sequences=True,
            time_major=False,
            go_backwards=True,
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
        nodes = tf.convert_to_tensor(features.nodes)
        query = self.token_embedding(tf.concat([type, nodes], axis=1))
        value = self.mask_embedding(tf.convert_to_tensor(features.mask))
        query = self.resnet(query)
        # # Add context to nodes
        # extract_window = 3
        # input = tf.reshape(nodes, shape=[1, batch_size, sequence_length, 1])
        # input = tf.image.extract_patches(
        #     images=input,
        #     sizes=[1, 1, extract_window, 1],
        #     strides=[1, 1, 1, 1],
        #     rates=[1, 1, 1, 1],
        #     padding="SAME",
        # )
        # input = tf.squeeze(input, axis=0)
        # value = self.token_embedding(tf.reverse(nodes, [1]))

        # Add context to each timesteps node vectors first
        in_state_h = tf.concat(features.rnn_state[0], axis=0)
        in_state_c = tf.concat(features.rnn_state[1], axis=0)
        query, state_h, state_c = self.nodes_lstm(
            query, initial_state=[in_state_h, in_state_c]
        )
        # Add backwards context to each timesteps node vectors first
        query, state_h, state_c = self.nodes_lstm_bwd(
            query, initial_state=[state_h, state_c]
        )

        # Reshape to [Time/Batch, 1, sequence * self.lstm_units]
        # time_in = self.time_compress(query)

        # + 2 is for the contatenated type information
        # context_vec_length = sequence_length + 2
        # time_in = tf.reshape(
        #     query, [batch_size, 1, context_vec_length * self.lstm_units]
        # )
        # time_in = self.time_compress(time_in)

        state_h = tf.tile(state_h[-1:], [sequence_length + 2, 1])
        state_c = tf.tile(state_c[-1:], [sequence_length + 2, 1])
        time_out, state_h, state_c = self.time_lstm(
            query, initial_state=[state_h, state_c]
        )
        # time_out = self.time_expand(time_out, sequence_length)
        time_out = self.attention([time_out, value])

        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        return (time_out, sequence_length)
