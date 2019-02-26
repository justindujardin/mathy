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
        Dense,
        RNN,
        SimpleRNNCell,
        Input,
        Embedding,
        LSTM,
    )
    from tensorflow.keras.experimental import SequenceFeatures
    from tensorflow_estimator.contrib.estimator.python import estimator
    from tensorflow.python.training import adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Concatenate, Flatten

    action_size = params["action_size"]
    learning_rate = params.get("learning_rate", 0.01)
    feature_columns = params["feature_columns"]
    sequence_columns = params["sequence_columns"]
    sequence_features_layer = SequenceFeatures(
        sequence_columns, name="sequence_features"
    )
    features_layer = DenseFeatures(feature_columns, name="input_features")
    sequence_features = {
        FEATURE_TOKEN_TYPES: features[FEATURE_TOKEN_TYPES],
        FEATURE_TOKEN_VALUES: features[FEATURE_TOKEN_VALUES],
    }
    sequence_layer_out, sequence_length = sequence_features_layer(sequence_features)
    context_layer_out = features_layer(features)
    for units in params["hidden_units"]:
        context_layer_out = Dense(units, activation=tf.nn.relu)(context_layer_out)

    sequence_layer_dense = Flatten()(sequence_layer_out)
    context_layer_dense = Flatten()(context_layer_out)
    network = Concatenate()([sequence_layer_dense, context_layer_dense])

    training = mode == tf.estimator.ModeKeys.TRAIN
    lstm_out = Flatten()(LSTM(32)(sequence_layer_out))
    focus_net = Dense(6)(lstm_out)
    value_net = Dense(12)(lstm_out)
    policy_net = Dense(action_size)(lstm_out)

    value_logits = Dense(1, activation="tanh", name="value_output")(value_net)
    policy_logits = Dense(action_size, activation="softmax", name="policy_output")(
        policy_net
    )
    focus_logits = Dense(1, activation="sigmoid", name="focus_output")(focus_net)
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
