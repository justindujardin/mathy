from typing import Dict
import tensorflow as tf

from ...core.expressions import MathTypeKeysMax
from ...features import (
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
from .bahdanau_attention import BahdanauAttention
from .resnet_stack import ResNetStack


class MathEmbedding(tf.keras.layers.Layer):
    def __init__(self, units: int = 128, **kwargs):
        super(MathEmbedding, self).__init__(**kwargs)
        self.units = units
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
            units, name="nodes_lstm", return_sequences=True, time_major=False
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
        self.f_moves_remaining = tf.feature_column.numeric_column(
            key=FEATURE_MOVES_REMAINING, dtype=tf.int64
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

    def call(self, features, initialize=False):
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
        initial_state = [self.dense_c(context_inputs), self.dense_h(context_inputs)]

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

        # Add context to each timesteps node vectors first
        outputs = self.nodes_lstm(outputs, initial_state=initial_state)

        # Apply LSTM to the output sequences to capture context across timesteps
        # in the batch.
        #
        # NOTE: This does process the observation timesteps across the batch axis with
        #
        # shape = [Observations, ObservationNodeVectors, self.units]
        time_out = self.time_lstm(outputs)
        # Convert context features into unit-sized layer
        context_vector, _ = self.attention(time_out, context_inputs)

        time_out = time_out + tf.expand_dims(context_vector, axis=1)

        return (context_vector, time_out, sequence_length)
