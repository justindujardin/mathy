import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy
from helper import normalized_columns_initializer


class AC_Network:
    def __init__(self, s_size, a_size, scope, trainer):
        self.s_size = s_size
        self.a_size = a_size
        self.scope = scope
        self.trainer = trainer
        with tf.variable_scope(scope):
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
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
        c_init = numpy.zeros((1, lstm_cell.state_size.c), numpy.float32)
        h_init = numpy.zeros((1, lstm_cell.state_size.h), numpy.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.imageIn)[:1]
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
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
