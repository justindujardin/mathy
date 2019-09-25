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
    def __init__(self, units: int = 128, burn_in_steps: int = 10, **kwargs):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.burn_in_steps = burn_in_steps
        self.init_rnn_state()
        self.attention = BahdanauAttention(self.units)
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.concat = tf.keras.layers.Concatenate(name=f"embedding_concat")
        self.input_dense = tf.keras.layers.Dense(
            self.units, name="embedding_input", use_bias=False
        )
        self.resnet = ResNetStack(units=units, name="embedding_resnet", num_layers=4)
        self.time_lstm = tf.keras.layers.LSTM(
            units,
            name="timestep_lstm",
            return_sequences=True,
            time_major=True,
            return_state=True,
        )
        self.nodes_lstm = tf.keras.layers.LSTM(
            units,
            name="nodes_lstm",
            return_sequences=True,
            time_major=False,
            return_state=True,
        )
        self.nodes_lstm_bwd = tf.keras.layers.LSTM(
            units,
            name="nodes_lstm_bwd",
            return_sequences=True,
            time_major=False,
            go_backwards=True,
            return_state=True,
        )
        self.embedding = tf.keras.layers.Dense(units=self.units, name="embedding")
        self.attention = BahdanauAttention(units, name="attention")
        # self.time_compress = tf.keras.layers.Dense(1, name="nodes_compress")
        # self.time_expand = tf.keras.layers.Dense(self.units, name="nodes_expand")

    def init_rnn_state(self):
        """WIP to train RNN state as in: https://openreview.net/pdf?id=r1lyTjAqYX

        - use a burn-in of the LSTM state from zero to get a good starting state
          before backproping into the LSTM. This is supposed to avoid the first
          few (really bad) predictions given by the zero-state from making harmful
          changes to the RNN's weights
        - store [(1, units), (1, units)] rnn state with each observation so that
          when using observations from the replay buffer we can mitigate
          representational drift. The intuition here is that the RNN state matters
          for observation playback, so it has to be stored with the observation.
          If you think about using the hidden state from a really good model to
          replay an old observation that has tracked its discounted reward value,
          the reward value will not be in agreement with the prediction based on
          the current RNN state. Using the RNN state from the observation timestep
          allows the state to propagate back to the very first zero-state of the
          agent.
        """
        self.state_c = tf.Variable(
            tf.zeros([1, self.units]),
            trainable=False,
            name="embedding/rnn/agent_state_c",
        )
        self.state_h = tf.Variable(
            tf.zeros([1, self.units]),
            trainable=False,
            name="embedding/rnn/agent_state_h",
        )
        self.burn_in_steps_counter = 0

    def reset_rnn_state(self):
        """Zero out the RNN state for a new episode"""
        self.state_c.assign(tf.zeros([1, self.units]))
        self.state_h.assign(tf.zeros([1, self.units]))
        self.burn_in_steps_counter = 0

    def call(
        self, features: MathyWindowObservation, do_burn_in=False
    ) -> Tuple[tf.Tensor, int]:
        batch_size = len(features.nodes)
        sequence_length = len(features.nodes[0])
        max_depth = MathTypeKeysMax
        batch_nodes = tf.one_hot(features.nodes, dtype=tf.float32, depth=max_depth)

        outputs = []
        for j, sequence_inputs in enumerate(batch_nodes):
            # Common resnet feature extraction
            example = self.resnet(self.input_dense(self.flatten(sequence_inputs)))
            outputs.append(tf.reshape(example, [sequence_length, -1]))

        # Process the sequence embeddings with a RNN to capture temporal
        # features. NOTE: this relies on the batches of observations being
        # passed to this function to be in chronological order of the batch
        # axis.
        # outputs is a tensor that's a sequence of observations at timesteps
        # where each observation has a sequence of nodes.
        #
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

        # time_out, _, _ = self.time_lstm(outputs)
        time_state = [
            tf.tile(state_h[-1:], [sequence_length, 1]),
            tf.tile(state_c[-1:], [sequence_length, 1]),
        ]
        time_out, state_h, state_c = self.time_lstm(outputs, initial_state=time_state)
        self.state_h.assign(state_h[-1:])
        self.state_c.assign(state_c[-1:])

        # For the first (n) steps, don't backprop into the RNN, just update
        # the stored initial state.
        #
        # TODO: This is way optimistic. It's not totally clear how to implement
        #       the burn-in procedure they described, but I think stopping the
        #       gradients should do it..?
        #
        #  Ref: https://openreview.net/pdf?id=r1lyTjAqYX
        # if self.burn_in_steps_counter <= self.burn_in_steps:
        #     self.burn_in_steps_counter += 1
        #     time_out = tf.stop_gradient(time_out)

        return (time_out, sequence_length)
