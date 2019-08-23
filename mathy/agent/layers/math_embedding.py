import tensorflow as tf

from ..features import (
    FEATURE_BWD_VECTORS,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_LAST_RULE,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
)
from ...core.expressions import MathTypeKeysMax


class MathEmbedding(tf.keras.layers.Layer):
    """Layer that takes in Tensorflow feature columns, and outputs a tuple of
    (context_embedding, sequence_embedding, sequence_length)"""

    def __init__(self, **kwargs):
        self.build_feature_columns()
        super(MathEmbedding, self).__init__(**kwargs)

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

        vocab_buckets = MathTypeKeysMax + 1

        #
        # Sequence features
        #
        # self.feat_policy_mask = tf.feature_column.sequence_numeric_column(
        #     key=FEATURE_MOVE_MASK, dtype=tf.int64, shape=self.action_size
        # )
        self.feat_bwd_vectors = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_identity(
                key=FEATURE_BWD_VECTORS, num_buckets=vocab_buckets
            ),
            dimension=32,
        )
        self.feat_fwd_vectors = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_identity(
                key=FEATURE_FWD_VECTORS, num_buckets=vocab_buckets
            ),
            dimension=32,
        )
        self.feat_last_bwd_vectors = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_identity(
                key=FEATURE_LAST_BWD_VECTORS, num_buckets=vocab_buckets
            ),
            dimension=32,
        )
        self.feat_last_fwd_vectors = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_identity(
                key=FEATURE_LAST_FWD_VECTORS, num_buckets=vocab_buckets
            ),
            dimension=32,
        )
        self.sequence_columns = [
            self.feat_fwd_vectors,
            self.feat_bwd_vectors,
            self.feat_last_fwd_vectors,
            self.feat_last_bwd_vectors,
        ]

        self.seq_extractor = tf.keras.experimental.SequenceFeatures(
            self.sequence_columns, name="seq_features"
        )
        self.ctx_extractor = tf.keras.layers.DenseFeatures(
            self.feature_columns, name="ctx_features"
        )

    def call(self, features):
        #
        # TODO: Verify that we're one-hot encoding the vectors here...
        # I suspect we need to be for the context vecotrs to be most efficient.
        #
        sequence_features = {
            FEATURE_BWD_VECTORS: features[FEATURE_BWD_VECTORS],
            FEATURE_FWD_VECTORS: features[FEATURE_FWD_VECTORS],
            FEATURE_LAST_BWD_VECTORS: features[FEATURE_LAST_BWD_VECTORS],
            FEATURE_LAST_FWD_VECTORS: features[FEATURE_LAST_FWD_VECTORS],
        }
        sequence_inputs, sequence_length = self.seq_extractor(sequence_features)
        context_inputs = self.ctx_extractor(features)
        return context_inputs, sequence_inputs, sequence_length
