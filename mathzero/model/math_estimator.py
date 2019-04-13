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
import numpy
import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    """Attention from: https://www.tensorflow.org/alpha/tutorials/sequences/image_captioning#model"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


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


def BiDirectionalLSTM(units, name="bi_lstm_stack", state=True):
    """Bi-directional LSTMs using Tensorflow's keras implementation"""
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Concatenate, Add

    def func(input_layer):
        forward_layer = LSTM(
            units, return_sequences=True, return_state=True, name="lstm/forward"
        )
        backward_layer = LSTM(
            units,
            return_sequences=True,
            go_backwards=True,
            return_state=True,
            name="lstm/backward",
        )
        lstm_fwd, state_h_fwd, state_c_fwd = forward_layer(input_layer)
        lstm_bwd, state_h_bwd, state_c_bwd = backward_layer(input_layer)

        return (
            Add(name=f"{name}_states")([state_h_fwd, state_h_bwd]),
            Concatenate(name=name)([lstm_fwd, lstm_bwd, input_layer]),
        )

    return func


from tensorflow.keras.layers import Layer, Flatten, Dense


class PredictSequences(Layer):
    def __init__(self, num_predictions=2, **kwargs):
        self.num_predictions = num_predictions
        self.activate = Dense(num_predictions, activation="softmax")
        self.cell_units = 4
        self.stack_height = 12
        self.dense_stack = [ResidualDenseLayer(self.cell_units)]
        for i in range(self.stack_height - 1):
            self.dense_stack.append(ResidualDenseLayer(self.cell_units))
        super(PredictSequences, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor):
        for layer in self.dense_stack:
            input_tensor = layer(input_tensor)
        return self.activate(input_tensor)


class SequenceContext(Layer):
    def __init__(self, input_context, batch_size, **kwargs):
        self.batch_size = batch_size
        self.input_context = input_context
        self.output_dim = 32
        super(SequenceContext, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.output_dim])

    def build(self, input_shape):
        # we resize the context_features on axis 1 (0 is batch) to match
        # the input layer so they can be concatenated.
        self.input_dense = Dense(self.output_dim, input_shape=(input_shape[0], None))
        self.context_dense = Dense(
            self.output_dim, input_shape=(self.input_context.get_shape()[0], None)
        )

    def call(self, input_tensor):
        from tensorflow.keras.layers import LSTM, Concatenate, Add, ZeroPadding1D

        # TODO: try this:
        #  - flatten input/context to 1d
        #  - pad to the longest of the input/context
        #  - combine, and enjoy!

        input_tensor = self.input_dense(input_tensor)
        input_tensor = tf.transpose(input_tensor)
        return input_tensor
        # context_tensor = self.context_dense(self.input_context)
        # context_tensor = tf.transpose(context_tensor)
        # npad is a tuple of (n_before, n_after) for each dimension
        # npad = ((None, self.batch_size), (self.batch_size, None))
        # b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)
        # size = tf.size(context_tensor)
        # batch = tf.constant(self.batch_size, dtype=tf.int32)
        # remainder = tf.reduce_max(tf.floormod(size, batch))
        # row_length = tf.cast(
        #     tf.math.ceil(tf.divide(tf.subtract(size, remainder), batch)), tf.int32
        # )
        # paddings = [[0, 0], [0, row_length]]
        # tf.pad is weird, it seems to do a t/b/l/r padding kind of like CSS?
        # this padding below extends all the context rows to the
        # context_tensor = tf.pad(context_tensor, paddings, "CONSTANT")
        # # after padding we can reshape it
        # context_tensor = tf.reshape(
        #     context_tensor, [self.batch_size, -1], name="context_reshape"
        # )

        # x1 = input_tensor  # Input(shape=(10, 100))
        # x2 = context_tensor  # Input(shape=(20, 30))
        # x2_unpadded = (
        #     ZeroPadding1D((0, x1.shape[2] - x2.shape[2]))(
        #         tf.reshape(x2, (-1, x2.shape[2], x2.shape[1]))
        #     ),
        # )

        # x2_padded = tf.reshape(x2_unpadded, (-1, x2.shape[1], x1.shape[2]))
        # x2_padded = tf.reshape(
        #     ZeroPadding1D((0, x1.shape[2] - x2.shape[2]))(
        #         K.reshape(x2, (-1, x2.shape[2], x2.shape[1]))
        #     ),
        #     (-1, x2.shape[1], x1.shape[2])
        # )
        # return Concatenate(name="sequence_context")([input_tensor, context_tensor])


def math_estimator(features, labels, mode, params):
    import tensorflow as tf
    from tensorflow.python.ops import init_ops
    from tensorflow.feature_column import make_parse_example_spec
    from tensorflow.keras.layers import (
        DenseFeatures,
        Dense,
        Concatenate,
        Input,
        TimeDistributed,
    )
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
    hidden_states, sequence_inputs = BiDirectionalLSTM(12)(sequence_inputs)

    #
    # Context input layers
    #
    features_layer = DenseFeatures(feature_columns, name="inputs/context")
    context_inputs = features_layer(features)
    # Concatenated context and sequence vectors
    with tf.compat.v1.variable_scope("residual_tower"):
        for i in range(6):
            context_inputs = ResidualDenseLayer(4)(context_inputs)

    predict_policy = TimeDistributed(PredictSequences(action_size))
    sequence_outputs = predict_policy(sequence_inputs)
    # sequence_outputs = tf.map_fn(predict_over_nodes, sequence_inputs)
    network = Concatenate(name="mixed_features")(
        [attention(sequence_outputs, 4096), context_inputs]
    )
    # Gather the last sequence output for value prediction
    value_logits = Dense(1, activation="tanh", name="value_logits")(network)
    # Flatten policy logits to 1d arrays
    policy_logits = sequence_outputs
    policy_shape = tf.shape(policy_logits)
    policy_logits = tf.reshape(
        policy_logits, [policy_shape[0], -1, 1], name="policy_reshape"
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
    # if labels is not None:
    # sess = tf.compat.v1.Session()
    # with sess.as_default():
    # print("POLICY LOGITS", policy_logits.eval())
    # print("POLICY LABELS", labels["policy"].eval())
    # print("VALUE LOGITS", value_logits.eval())
    # print("VALUE LABELS", labels["policy"].eval())
    # Multi-task prediction heads
    policy_head = estimator.head.regression_head(name="policy", label_dimension=1)
    value_head = estimator.head.regression_head(name="value", label_dimension=1)
    multi_head = estimator.multi_head.multi_head([policy_head, value_head])
    return multi_head.create_estimator_spec(features, mode, logits, labels, optimizer)
