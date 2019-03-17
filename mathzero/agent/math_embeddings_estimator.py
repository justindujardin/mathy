import tensorflow as tf
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    DenseFeatures,
    Flatten,
    Input,
)
from tensorflow_estimator.contrib.estimator.python import estimator
from ..environment_state import to_sparse_tensor
from ..model.features import (
    FEATURE_NODE_COUNT,
    FEATURE_TOKEN_TYPES,
    FEATURE_TOKEN_VALUES,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_PROBLEM_TYPE,
    TRAIN_LABELS_AS_MATRIX,
    TRAIN_LABELS_TARGET_FOCUS,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_REWARD,
)
from ..model.math_attention import attention
from tensorflow.keras.layers import Layer


def dense_with_activation(input, units, name):
    with tf.compat.v1.variable_scope(name):
        activate = Activation("relu")
        normalize = BatchNormalization()
        dense = Dense(units, use_bias=False)
        return activate(normalize(dense(input)))


def BiDirectionalLSTM(units, name="bi_lstm_stack"):
    """Bi-directional stacked LSTMs using Tensorflow's keras implementation"""
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Concatenate

    # On the second stacked LSTM we reduce the dimensionality by half, because... I felt like it
    # TODO: Investigate whether this is a good idea?
    half_units = int(units / 2)

    def func(input_layer):
        rnn_fwd1 = LSTM(units, return_sequences=True, name="fwd_lstm")(input_layer)
        rnn_fwd2 = LSTM(half_units, return_sequences=True, name="fwd_lstm")(rnn_fwd1)
        rnn_bwd1 = LSTM(
            units, return_sequences=True, go_backwards=True, name="bwd_lstm"
        )(input_layer)
        rnn_bwd2 = LSTM(
            half_units, return_sequences=True, go_backwards=True, name="bwd_lstm"
        )(rnn_bwd1)
        return Concatenate(name=name)([rnn_fwd2, rnn_bwd2, input_layer])

    return func


def embeddings_estimator(features, labels, mode, params):
    action_size = params["action_size"]
    embedding_dimensions = params["embedding_dimensions"]
    sequence_columns = params["sequence_columns"]
    sequence_features = {
        FEATURE_TOKEN_TYPES: features[FEATURE_TOKEN_TYPES],
        FEATURE_TOKEN_VALUES: features[FEATURE_TOKEN_VALUES],
    }
    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="sequence_features"
    )(sequence_features)

    lstm = BiDirectionalLSTM(128)(sequence_inputs)
    sequence_vectors = attention(lstm, 16)

    #
    # Non-sequence input layers
    #
    feature_columns = params["feature_columns"]
    features_layer = DenseFeatures(feature_columns, name="input_features")
    context_inputs = features_layer(features)
    context_inputs = dense_with_activation(context_inputs, 8, "context_embed")

    # Concatenated context and sequence vectors
    embedding = Concatenate(name="math_vectors")([context_inputs, sequence_vectors])
    embedding = Dense(embedding_dimensions)(embedding)
    embedding = tf.cast(embedding, tf.float32)
    # Compute predictions.
    assert mode == tf.estimator.ModeKeys.PREDICT
    predictions = {"embedding": embedding}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def math_embeddings_network(action_size, embedding_dimensions=32):
    class MathEmbeddingLayer(Layer):
        def __init__(self, **kwargs):
            self.action_size = action_size
            self.embedding_dimensions = embedding_dimensions
            super(MathEmbeddingLayer, self).__init__(**kwargs)

        def call(self, inputs):
            #
            # Sequence features
            #
            f_token_types_sequence = tf.feature_column.embedding_column(
                tf.feature_column.sequence_categorical_column_with_hash_bucket(
                    key=FEATURE_TOKEN_TYPES, hash_bucket_size=24, dtype=tf.int8
                ),
                dimension=32,
            )
            f_token_values_sequence = tf.feature_column.embedding_column(
                tf.feature_column.sequence_categorical_column_with_hash_bucket(
                    key=FEATURE_TOKEN_VALUES, hash_bucket_size=128, dtype=tf.string
                ),
                dimension=32,
            )
            sequence_features = {
                FEATURE_TOKEN_TYPES: inputs[FEATURE_TOKEN_TYPES],
                # FEATURE_TOKEN_VALUES: inputs[FEATURE_TOKEN_VALUES],
            }
            sequence_columns = [
                f_token_types_sequence,
                # f_token_values_sequence
            ]
            sequence_inputs, sequence_length = SequenceFeatures(
                sequence_columns, name="sequence_features"
            )(sequence_features)

            lstm = BiDirectionalLSTM(128)(sequence_inputs)
            sequence_vectors = attention(lstm, 16)

            #
            # Non-sequence features/layers
            #
            f_move_count = tf.feature_column.numeric_column(
                key=FEATURE_MOVE_COUNTER, dtype=tf.int16
            )
            f_moves_remaining = tf.feature_column.numeric_column(
                key=FEATURE_MOVES_REMAINING, dtype=tf.int16
            )
            f_node_count = tf.feature_column.numeric_column(
                key=FEATURE_NODE_COUNT, dtype=tf.int16
            )
            f_problem_type = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    key=FEATURE_PROBLEM_TYPE, num_buckets=32
                )
            )
            feature_columns = [
                f_problem_type,
                f_node_count,
                f_move_count,
                f_moves_remaining,
            ]
            features_layer = DenseFeatures(feature_columns, name="input_features")
            context_inputs = features_layer(inputs)
            context_inputs = dense_with_activation(context_inputs, 8, "context_embed")

            # Concatenated context and sequence vectors
            embedding = Concatenate(name="math_vectors")(
                [context_inputs, sequence_vectors]
            )
            embedding = Dense(embedding_dimensions)(embedding)
            return embedding

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.embedding_dimensions)

    return MathEmbeddingLayer()
