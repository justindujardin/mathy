from ..model.features import (
    FEATURE_NODE_COUNT,
    FEATURE_FWD_VECTORS,
    FEATURE_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
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

    def func(input_layer):
        forward = LSTM(units, return_sequences=True, name="lstm/forward")(input_layer)
        backward = LSTM(
            units, return_sequences=True, go_backwards=True, name="lstm/backward"
        )(input_layer)
        return Concatenate(name=name)([forward, backward, input_layer])

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
    learning_rate = params.get("learning_rate", 0.01)
    training = mode == tf.estimator.ModeKeys.TRAIN

    #
    # Sequential feature layers
    #
    sequence_features = {
        FEATURE_BWD_VECTORS: features[FEATURE_BWD_VECTORS],
        FEATURE_FWD_VECTORS: features[FEATURE_FWD_VECTORS],
        FEATURE_LAST_BWD_VECTORS: features[FEATURE_LAST_BWD_VECTORS],
        FEATURE_LAST_FWD_VECTORS: features[FEATURE_LAST_FWD_VECTORS],
    }
    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="inputs/sequence"
    )(sequence_features)
    sequence_inputs = BiDirectionalLSTM(128)(sequence_inputs)
    sequence_inputs = attention(sequence_inputs, 2048, name="inputs/sequence_attention")

    #
    # Context input layers
    #
    features_layer = DenseFeatures(feature_columns, name="inputs/context")
    context_inputs = features_layer(features)
    # Concatenated context and sequence vectors
    network = Concatenate(name="inputs/concat")([context_inputs, sequence_inputs])

    with tf.compat.v1.variable_scope("residual_tower"):
        for i in range(40):
            network = ResidualDenseLayer(4)(network)

    # Concatenated context and sequence vectors
    value_logits = Dense(1, activation="tanh", name="value_logits")(network)
    policy_logits = Dense(action_size, activation="softmax", name="policy_logits")(
        network
    )
    logits = {"policy": policy_logits, "value": value_logits}

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    # output histograms for all trainable variables.
    summary_interval = 50
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.name, var)

    with tf.compat.v1.variable_scope("stats"):

        # Output values
        tf.compat.v1.summary.scalar("value/mean", tf.reduce_mean(value_logits))
        tf.compat.v1.summary.scalar(
            "value/variance", tf.math.reduce_variance(value_logits)
        )
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