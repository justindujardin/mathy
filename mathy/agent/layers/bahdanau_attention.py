import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    """Attention from: https://www.tensorflow.org/alpha/tutorials/sequences/image_captioning#model"""

    def __init__(self, units, name="attention"):
        super(BahdanauAttention, self).__init__()
        with tf.compat.v1.variable_scope(name):
            self.W1 = tf.keras.layers.Dense(units, name=f"{name}/w1")
            self.W2 = tf.keras.layers.Dense(units, name=f"{name}/w2")
            self.V = tf.keras.layers.Dense(1, name=f"{name}/v")

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

