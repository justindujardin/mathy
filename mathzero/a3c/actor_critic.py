import numpy
import tensorflow as tf
from tensorflow.feature_column import make_parse_example_spec
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    DenseFeatures,
    Embedding,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.python.ops import init_ops
from tensorflow.python.training import adam
from tensorflow_estimator.contrib.estimator.python import estimator

from helper import normalized_columns_initializer

from ..environment_state import to_sparse_tensor
from .features import (
    FEATURE_NODE_COUNT,
    FEATURE_TOKEN_TYPES,
    FEATURE_TOKEN_VALUES,
    TRAIN_LABELS_AS_MATRIX,
    TRAIN_LABELS_TARGET_FOCUS,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_REWARD,
)
from .math_attention import attention


def BiDirectionalLSTM(units, name="bi_lstm_stack"):
    """Bi-directional stacked LSTM using Tensorflow's keras implementation"""

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


def dense_with_activation(input, units, name):
    with tf.compat.v1.variable_scope(name):
        activate = Activation("relu")
        normalize = BatchNormalization()
        dense = Dense(units, use_bias=False)
        return activate(normalize(dense(input)))


class A3CModel:
    """Async Actor Critic network"""

    def __init__(self, s_size, a_size, scope, trainer):
        self.s_size = s_size
        self.a_size = a_size
        self.scope = scope
        self.trainer = trainer
        with tf.compat.v1.variable_scope(scope):
            self.build_network()

    def build_network(self):
        # Input and visual encoding layers
        self.inputs = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
        self.conv1 = slim.conv2d(
            activation_fn=tf.nn.elu,
            inputs=self.imageIn,
            num_outputs=16,
            kernel_size=[8, 8],
            stride=[4, 4],
            padding="VALID",
        )
        self.conv2 = slim.conv2d(
            activation_fn=tf.nn.elu,
            inputs=self.conv1,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[2, 2],
            padding="VALID",
        )
        hidden = slim.fully_connected(
            slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu
        )

        # Recurrent network for temporal dependencies
        lstm_cell = tf.keras.layers.LSTMCell(256, state_is_tuple=True)
        c_init = numpy.zeros((1, lstm_cell.state_size.c), numpy.float32)
        h_init = numpy.zeros((1, lstm_cell.state_size.h), numpy.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.imageIn)[:1]
        state_in = tf.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.keras.layers.RNN(
            lstm_cell,
            rnn_in,
            initial_state=state_in,
            sequence_length=step_size,
            time_major=False,
        )
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(
            rnn_out,
            self.a_size,
            activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=None,
        )
        self.value = slim.fully_connected(
            rnn_out,
            1,
            activation_fn=None,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None,
        )

        # Only the worker network need ops for loss functions and gradient updating.
        if self.scope != "global":
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, self.a_size, dtype=tf.float32
            )
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1]
            )

            # Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(
                tf.square(self.target_v - tf.reshape(self.value, [-1]))
            )
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(
                tf.log(self.responsible_outputs) * self.advantages
            )
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global")
            self.apply_grads = self.trainer.apply_gradients(zip(grads, global_vars))


def math_estimator(features, labels, mode, params):

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
