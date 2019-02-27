from .features import (
    FEATURE_NODE_COUNT,
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_REWARD,
    TRAIN_LABELS_TARGET_FOCUS,
    TRAIN_LABELS_AS_MATRIX,
)
from ..environment_state import to_sparse_tensor


def math_estimator(features, labels, mode, params):
    import tensorflow as tf
    from tensorflow.python.ops import init_ops
    from tensorflow.feature_column import make_parse_example_spec
    from tensorflow.keras.layers import (
        DenseFeatures,
        Activation,
        Dense,
        RNN,
        SimpleRNNCell,
        Input,
        Embedding,
        LSTM,
        Concatenate,
        Flatten,
        BatchNormalization,
    )
    from tensorflow.keras.experimental import SequenceFeatures
    from tensorflow_estimator.contrib.estimator.python import estimator
    from tensorflow.python.training import adam
    from tensorflow.keras.models import Model

    def dense_with_activation(input, units=32):
        activate = Activation("relu")
        normalize = BatchNormalization()
        dense = Dense(units, use_bias=False)
        return activate(normalize(dense(input)))

    action_size = params["action_size"]
    learning_rate = params.get("learning_rate", 0.01)
    training = mode == tf.estimator.ModeKeys.TRAIN

    sequence_columns = params["sequence_columns"]
    sequence_features = {
        FEATURE_TOKEN_TYPES: features[FEATURE_TOKEN_TYPES],
        FEATURE_TOKEN_VALUES: features[FEATURE_TOKEN_VALUES],
    }
    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="sequence_features"
    )(sequence_features)
    lstm_out = LSTM(3, return_sequences=True)(sequence_inputs)
    lstm_last = lstm_out[:, -1]
    lstm_last = dense_with_activation(lstm_last, 6)

    #
    # Non-sequence input layers
    #
    feature_columns = params["feature_columns"]
    features_layer = DenseFeatures(feature_columns, name="input_features")
    context_inputs = features_layer(features)
    context_inputs = dense_with_activation(context_inputs)

    # Concatenated context and sequence vectors
    network = Concatenate()([context_inputs, lstm_last])
    for units in params["hidden_units"]:
        network = dense_with_activation(network, 2)

    value_logits = Dense(1, activation="tanh", name="value_output")(network)
    policy_logits = Dense(action_size, activation="softmax", name="policy_output")(
        network
    )
    focus_logits = Dense(1, activation="sigmoid", name="focus_output")(network)
    logits = {"policy": policy_logits, "value": value_logits, "focus": focus_logits}

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    # Model prediction heads
    policy_head = estimator.head.regression_head(
        name="policy", label_dimension=action_size
    )
    focus_head = estimator.head.regression_head(name="focus", label_dimension=1)
    value_head = estimator.head.regression_head(name="value", label_dimension=1)
    multi_head = estimator.multi_head.multi_head([policy_head, value_head, focus_head])
    return multi_head.create_estimator_spec(features, mode, logits, labels, optimizer)
