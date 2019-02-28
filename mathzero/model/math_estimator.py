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
from .math_attention import attention


def math_estimator(features, labels, mode, params):
    import tensorflow as tf
    from tensorflow.python.ops import init_ops
    from tensorflow.feature_column import make_parse_example_spec
    from tensorflow.keras.layers import (
        DenseFeatures,
        Activation,
        Dense,
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

    def dense_with_activation(input, units, name):
        with tf.compat.v1.variable_scope(name):
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

    def BiDirectionalLSTM(lstm_units):
        def func(input_):
            rnn_fwd1 = LSTM(lstm_units, return_sequences=True)(input_)
            rnn_bwd1 = LSTM(lstm_units, return_sequences=True, go_backwards=True)(
                input_
            )
            rnn_bidir1 = Concatenate(name="bi_rnn_out")([rnn_fwd1, rnn_bwd1])
            return rnn_bidir1

        return func

    with tf.name_scope("encode_attend_sequences"):
        lstm = LSTM(32, return_sequences=True)(sequence_inputs)
        lstm = LSTM(24, return_sequences=True)(lstm)
        lstm = BiDirectionalLSTM(12)(sequence_inputs)
        sequence_vectors = attention(lstm, 16)

    #
    # Non-sequence input layers
    #
    feature_columns = params["feature_columns"]
    features_layer = DenseFeatures(feature_columns, name="input_features")
    context_inputs = features_layer(features)
    context_inputs = dense_with_activation(context_inputs, 4, "context_vectors")

    # Concatenated context and sequence vectors
    network = Concatenate(name="math_vectors")([context_inputs, sequence_vectors])

    focus_logits = Dense(1, activation="sigmoid", name="focus_output")(network)
    value_logits = Dense(1, activation="tanh", name="value_output")(network)
    policy_logits = Dense(action_size, activation="softmax", name="policy_output")(
        network
    )
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
