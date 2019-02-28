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


def BiDirectionalLSTM(units, name="bi_lstm_stack"):
    """Bi-directional stacked LSTM using Tensorflow's keras implementation"""
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Concatenate

    # On the second stacked LSTM we reduce the dimensionality by half, because... I
    # felt like it might be cool because some CNNs use it to great effect.
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
        return Concatenate(name=name)([rnn_fwd2, rnn_bwd2])

    return func


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
    network = Concatenate(name="math_vectors")([context_inputs, sequence_vectors])

    value_logits = Dense(1, activation="tanh", name="value_logits")(network)
    policy_logits = Dense(action_size, activation="softmax", name="policy_logits")(
        network
    )
    focus_logits = Dense(1, activation="sigmoid", name="focus_logits")(network)
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
