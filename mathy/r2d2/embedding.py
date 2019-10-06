from typing import Dict, Tuple
import tensorflow as tf

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
from ..agent.layers.bahdanau_attention import BahdanauAttention
from ..agent.layers.resnet_stack import ResNetStack
from ..state import MathyBatchObservation, MathyObservation


class MathyEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 128,
        sequence_steps: int = 3,
        burn_in_steps: int = 10,
        **kwargs,
    ):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.units = units
        self.burn_in_steps = burn_in_steps
        self.sequence_steps = sequence_steps
        self.init_rnn_state(self.units)
        self.attention = BahdanauAttention(self.units)
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.concat = tf.keras.layers.Concatenate(name=f"embedding_concat")
        self.input_dense = tf.keras.layers.Dense(
            self.units, name="embedding_input", use_bias=False
        )
        self.resnet = ResNetStack(units=units, name="embedding_resnet", num_layers=4)
        self.time_lstm = tf.keras.layers.LSTM(
            units, name="timestep_lstm", return_sequences=True, time_major=True
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

    def init_rnn_state(self, units: int):
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
            tf.zeros([self.sequence_steps, units]),
            trainable=False,
            name="embedding/rnn/agent_state_c",
        )
        self.state_h = tf.Variable(
            tf.zeros([self.sequence_steps, units]),
            trainable=False,
            name="embedding/rnn/agent_state_h",
        )
        self.burn_in_steps_counter = 0

    def call(
        self, features: MathyBatchObservation, initial_state=None
    ) -> Tuple[tf.Tensor, int]:
        batch_size = len(features.nodes)
        sequence_length = len(features.nodes[0][0])
        max_depth = MathTypeKeysMax
        batch_nodes = tf.one_hot(features.nodes, dtype=tf.float32, depth=max_depth)

        outputs = []

        # Convert n-step batch inputs into a single-step
        # preserving the 0 (batch) and 1 (time) dimensions
        for i, n_step_batch in enumerate(batch_nodes):
            # If NOT given a stored initial state
            if initial_state is None:
                # Use the LSTM local models recurrent state
                n_step_state = [self.state_h, self.state_c]
            else:
                n_step_state = initial_state[i]

            batch_outs = []
            for j, sequence_inputs in enumerate(n_step_batch):
                # Common resnet feature extraction
                example = self.resnet(self.input_dense(self.flatten(sequence_inputs)))
                batch_outs.append(tf.reshape(example, [sequence_length, -1]))

            # Process the sequence embeddings with a RNN to capture temporal
            # features. NOTE: this relies on the batches of observations being
            # passed to this function to be in chronological order of the batch
            # axis.
            # batch_outs is a tensor that's a sequence of observations at timesteps
            # where each observation has a sequence of nodes.
            #
            # shape = [Observations, ObservationNodeVectors, self.units]
            batch_outs = tf.convert_to_tensor(batch_outs)

            # Add context to each timesteps node vectors first
            batch_outs, state_h, state_c = self.nodes_lstm(
                batch_outs, initial_state=n_step_state
            )
            # Add backwards context to each timesteps node vectors first
            batch_outs, state_h, state_c = self.nodes_lstm_bwd(
                batch_outs, initial_state=[state_h, state_c]
            )

            # Apply LSTM to the output sequences to capture context across timesteps
            # in the batch.
            #
            # NOTE: This does process the observation timesteps across the batch axis
            #
            # shape = [Observations, 1, ObservationNodeVectors * self.units]

            # reshape axis 1 to 1d so it corresponds to the expected
            # time_out = tf.reshape(batch_outs, [self.sequence_steps, 1, -1])

            time_out = self.time_lstm(batch_outs[-1::])

            # reshape with nodes
            # time_out = tf.reshape(time_out, [self.sequence_steps, outputs.shape[1], -1])

            self.state_c.assign(state_c)
            self.state_h.assign(state_h)

            outputs.append(time_out)

        outputs = tf.convert_to_tensor(outputs)

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

        return (outputs, sequence_length)
