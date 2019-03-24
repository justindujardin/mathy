from ..model.features import (
    FEATURE_NODE_COUNT,
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_LAST_TOKEN_VALUES,
    FEATURE_LAST_TOKEN_TYPES,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_VALUE,
    TRAIN_LABELS_AS_MATRIX,
)
from ..environment_state import to_sparse_tensor
from ..model.math_attention import attention


def ResidualDenseLayer(units, name="residual_block"):
    """Dense layer with residual input concatenation for help with backpropagation"""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Activation,
        Dense,
        Concatenate,
        BatchNormalization,
    )

    def func(input_layer):
        activate = Activation("relu")
        normalize = BatchNormalization()
        dense = Dense(units, use_bias=False)
        output = activate(normalize(dense(input_layer)))
        return Concatenate(name=name)([output, input_layer])

    return func


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


def math_estimator(features, labels, mode, params):
    import tensorflow as tf
    from tensorflow.python.ops import init_ops
    from tensorflow.feature_column import make_parse_example_spec
    from tensorflow.keras.layers import DenseFeatures, Dense, Concatenate
    from tensorflow.keras.experimental import SequenceFeatures
    from tensorflow_estimator.contrib.estimator.python import estimator
    from tensorflow.python.training import adam

    sequence_columns = params["sequence_columns"]
    feature_columns = params["feature_columns"]
    action_size = params["action_size"]
    embedding_dimensions = params["embedding_dimensions"]
    learning_rate = params.get("learning_rate", 0.01)
    training = mode == tf.estimator.ModeKeys.TRAIN

    #
    # Sequential feature layers
    #
    sequence_features = {
        FEATURE_TOKEN_TYPES: features[FEATURE_TOKEN_TYPES],
        FEATURE_TOKEN_VALUES: features[FEATURE_TOKEN_VALUES],
        FEATURE_LAST_TOKEN_TYPES: features[FEATURE_LAST_TOKEN_TYPES],
        FEATURE_LAST_TOKEN_VALUES: features[FEATURE_LAST_TOKEN_VALUES],
    }
    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="seq_features"
    )(sequence_features)
    sequence_inputs = BiDirectionalLSTM(128)(sequence_inputs)
    with tf.compat.v1.variable_scope("seq_residual_tower"):
        for i in range(12):
            sequence_inputs = ResidualDenseLayer(4)(sequence_inputs)
    # Use an attention layer to focus on parts
    sequence_inputs = attention(sequence_inputs, 64, name="tower_attn")

    #
    # Context input layers
    #
    features_layer = DenseFeatures(feature_columns, name="ctx_features")
    context_inputs = features_layer(features)
    # context residual tower
    with tf.compat.v1.variable_scope("ctx_residual_tower"):
        for i in range(8):
            context_inputs = ResidualDenseLayer(4)(context_inputs)

    # Concatenated context and sequence vectors
    network = Concatenate(name="towers_concat")([context_inputs, sequence_inputs])

    # Concatenated context and sequence vectors
    embedding_logits = Dense(embedding_dimensions, name="embedding_logits")(network)
    value_logits = Dense(1, activation="tanh", name="value_logits")(network)
    policy_logits = Dense(action_size, activation="softmax", name="policy_logits")(
        network
    )
    logits = {
        "policy": policy_logits,
        "value": value_logits,
        "embedding": embedding_logits,
    }

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    with tf.compat.v1.variable_scope("stats"):

        # Output values
        tf.compat.v1.summary.scalar("value/mean", tf.reduce_mean(value_logits))
        tf.compat.v1.summary.scalar(
            "value/variance", tf.math.reduce_variance(value_logits)
        )
        tf.compat.v1.summary.histogram("value/logits", value_logits)
        tf.compat.v1.summary.histogram("policy/logits", policy_logits)
        tf.compat.v1.summary.histogram("embeddings/logits", embedding_logits)

        # Training targets
        if labels is not None:
            tf.compat.v1.summary.scalar(
                "value/target_mean", tf.reduce_mean(labels[TRAIN_LABELS_TARGET_VALUE])
            )
            tf.compat.v1.summary.scalar(
                "value/target_variance",
                tf.math.reduce_variance(labels[TRAIN_LABELS_TARGET_VALUE]),
            )
            tf.compat.v1.summary.histogram(
                "policy/target", labels[TRAIN_LABELS_TARGET_PI]
            )
    # Multi-task prediction heads
    policy_head = estimator.head.regression_head(
        name="policy", label_dimension=action_size
    )
    value_head = estimator.head.regression_head(name="value", label_dimension=1)
    multi_head = estimator.multi_head.multi_head([policy_head, value_head])
    return multi_head.create_estimator_spec(features, mode, logits, labels, optimizer)
