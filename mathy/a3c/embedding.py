from typing import Any, Dict, Tuple

import tensorflow as tf

from ..agent.layers.bahdanau_attention import BahdanauAttention
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
        self.attention = BahdanauAttention(self.units)
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.concat = tf.keras.layers.Concatenate(name=f"embedding_concat")
        self.input_dense = tf.keras.layers.Dense(
            self.units, name="embedding_input", use_bias=False
        )
        self.resnet = ResNetStack(
            units=units // 10, name="embedding_resnet", num_layers=10
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
        self.nodes_lstm_bwd = tf.keras.layers.LSTM(
            self.lstm_units,
            name="nodes_lstm_bwd",
            return_sequences=True,
            time_major=False,
            go_backwards=True,
            return_state=True,
        )
        self.embedding = tf.keras.layers.Dense(units=self.units, name="embedding")
        self.attention = BahdanauAttention(units, name="attention")

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
        max_depth = MathTypeKeysMax
        nodes = tf.convert_to_tensor(features.nodes)
        # Add context to nodes
        extract_window = 3
        input = tf.reshape(nodes, shape=[1, batch_size, sequence_length, 1])
        input = tf.image.extract_patches(
            images=input,
            sizes=[1, 1, extract_window, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        nodes = tf.squeeze(input, axis=0)
        batch_nodes = tf.one_hot(nodes, dtype=tf.float32, depth=max_depth)

        outputs = []
        for j, sequence_inputs in enumerate(batch_nodes):
            example = self.resnet(self.input_dense(self.flatten(sequence_inputs)))
            outputs.append(tf.reshape(example, [sequence_length, -1]))

        # shape = [Observations, ObservationNodeVectors, self.units]
        outputs = tf.convert_to_tensor(outputs)

        # Add context to each timesteps node vectors first
        in_state_h = tf.concat(features.rnn_state[0], axis=0)
        in_state_c = tf.concat(features.rnn_state[1], axis=0)
        outputs, state_h, state_c = self.nodes_lstm(
            outputs, initial_state=[in_state_h, in_state_c]
        )
        # Add backwards context to each timesteps node vectors first
        outputs, state_h, state_c = self.nodes_lstm_bwd(
            outputs, initial_state=[state_h, state_c]
        )

        time_out, _, _ = self.time_lstm(outputs)
        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        return (time_out, sequence_length)
