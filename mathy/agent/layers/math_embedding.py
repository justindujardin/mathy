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
from .lstm import LSTM
from .keras_self_attention.seq_self_attention import SeqSelfAttention


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
        self.lstm = tf.keras.layers.LSTM(
            units,
            name="embedding_lstm",
            return_sequences=True,
            time_major=True,
            return_state=True,
        )

        self.embedding = tf.keras.layers.Dense(
            units=self.units, name="embedding", activation="relu"
        )
        self.build_feature_columns()
        # self.self_attention = SeqSelfAttention()

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
        self.feature_columns = [
            self.f_problem_type,
            self.f_last_rule,
            self.f_node_count,
            self.f_move_count,
            # self.f_moves_remaining,
        ]

        self.ctx_extractor = tf.keras.layers.DenseFeatures(
            self.feature_columns, name="ctx_features"
        )

    def call(self, features):
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
            tf.one_hot(
                features[FEATURE_LAST_BWD_VECTORS], dtype=tf.float32, depth=max_depth
            ),
            tf.one_hot(
                features[FEATURE_LAST_FWD_VECTORS], dtype=tf.float32, depth=max_depth
            ),
        ]
        batch_sequence_features = self.concat(batch_seq_one_hots)

        outputs = []
        # Convert one-hot to dense layer representations while
        # preserving the batch[0] and time[1] dimensions
        for sequence_inputs in batch_sequence_features:
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

        if batch_size > 1:
            # print("cool beans")
            pass

        # Apply LSTM to the output sequences to capture context across timesteps
        # in the batch.
        #
        # NOTE: This does process the observation timesteps across the batch axis with
        #
        # shape = [Observations, ObservationNodeVectors, self.units]
        time_out, state_c, state_h = self.lstm(outputs)

        # Bahdanau Attn
        attention_context, attention_weights = self.attention(time_out, context_inputs)
        return (attention_context, time_out, sequence_length)

        # # Self Attn
        # time_out = self.self_attention(time_out)
        # return (context_inputs, time_out, sequence_length)
