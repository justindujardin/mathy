from typing import Dict
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
        self.build_feature_columns()
        self.attention = BahdanauAttention(units, name="attention")
        self.dense_h = tf.keras.layers.Dense(units, name=f"lstm_initial_h")
        self.dense_c = tf.keras.layers.Dense(units, name=f"lstm_initial_c")

    def build_feature_columns(self):
        """Build out the Tensorflow Feature Columns that define the inputs from Mathy
        into the neural network."""
        self.f_move_count = tf.feature_column.numeric_column(
            key=FEATURE_MOVE_COUNTER, dtype=tf.int64
        )
        self.f_last_rule = tf.feature_column.numeric_column(
            key=FEATURE_LAST_RULE, dtype=tf.int64
        )
        self.f_node_count = tf.feature_column.numeric_column(
            key=FEATURE_NODE_COUNT, dtype=tf.int64
        )
        self.f_problem_type = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=FEATURE_PROBLEM_TYPE, num_buckets=32
            )
        )
        self.feature_columns = [self.f_last_rule, self.f_node_count, self.f_move_count]

        self.ctx_extractor = tf.keras.layers.DenseFeatures(
            self.feature_columns, name="ctx_features"
        )

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

    def call(self, features, initialize=False, initial_state=None):
        context_inputs = self.ctx_extractor(features)
        batch_size = len(features[FEATURE_BWD_VECTORS])
        sequence_length = len(features[FEATURE_BWD_VECTORS][0])
        max_depth = MathTypeKeysMax
        batch_seq_one_hots = [
            tf.one_hot(
                features[FEATURE_BWD_VECTORS], dtype=tf.float32, depth=max_depth
            ),
            tf.one_hot(
                features[FEATURE_FWD_VECTORS], dtype=tf.float32, depth=max_depth
            ),
        ]
        batch_sequence_features = self.concat(batch_seq_one_hots)

        outputs = []

        # Convert one-hot to dense layer representations while
        # preserving the 0 (batch) and 1 (time) dimensions
        for i, sequence_inputs in enumerate(batch_sequence_features):
            # Common resnet feature extraction
            example = self.resnet(self.input_dense(self.flatten(sequence_inputs)))
            outputs.append(tf.reshape(example, [sequence_length, -1]))

        # Build initial LSTM state conditioned on the context features
        # initial_state = [self.dense_c(context_inputs), self.dense_h(context_inputs)]

        # Process the sequence embeddings with a RNN to capture temporal
        # features. NOTE: this relies on the batches of observations being
        # passed to this function to be in chronological order of the batch
        # axis.
        # outputs is a tensor that's a sequence of observations at timesteps
        # where each observation has a sequence of nodes.
        #
        # shape = [Observations, ObservationNodeVectors, self.units]
        outputs = tf.convert_to_tensor(outputs)

        if batch_size > 1:
            # print("cool beans")
            pass

        # If NOT given a stored initial state
        if initial_state is None:
            # Use the LSTM local models recurrent state
            initial_state = [self.state_h, self.state_c]
        # Add context to each timesteps node vectors first
        outputs, state_h, state_c = self.nodes_lstm(
            outputs, initial_state=initial_state
        )
        # Add backwards context to each timesteps node vectors first
        outputs, state_h, state_c = self.nodes_lstm_bwd(
            outputs, initial_state=[state_h, state_c]
        )

        # Apply LSTM to the output sequences to capture context across timesteps
        # in the batch.
        #
        # NOTE: This does process the observation timesteps across the batch axis with
        #
        # shape = [Observations, 1, ObservationNodeVectors * self.units]

        # reshape axis 1 to 1d so it corresponds to the expected
        # time_out = tf.reshape(outputs, [self.sequence_steps, 1, -1])

        time_out = self.time_lstm(outputs)

        # reshape with nodes
        # time_out = tf.reshape(time_out, [self.sequence_steps, outputs.shape[1], -1])

        self.state_c.assign(state_c)
        self.state_h.assign(state_h)

        # Convert context features into unit-sized layer
        context_vector, _ = self.attention(time_out, context_inputs)

        time_out = time_out + tf.expand_dims(context_vector, axis=1)

        # For the first (n) steps, don't backprop into the RNN, just update
        # the stored initial state.
        #
        # TODO: This is way optimistic. It's not totally clear how to implement
        #       the burn-in procedure they described, but I think stopping the
        #       gradients should do it..?
        #
        #  Ref: https://openreview.net/pdf?id=r1lyTjAqYX
        if self.burn_in_steps_counter <= self.burn_in_steps:
            self.burn_in_steps_counter += 1
            context_vector = tf.stop_gradient(context_vector)
            time_out = tf.stop_gradient(time_out)

        return (context_vector, time_out, sequence_length)
