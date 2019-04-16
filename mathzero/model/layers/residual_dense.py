import tensorflow as tf


def ResidualDense(units, name="residual_block"):
    """Dense layer with residual input concatenation for help with backpropagation"""

    def func(input_layer):
        activate = tf.keras.layers.Activation("relu")
        normalize = tf.keras.layers.BatchNormalization()
        dense = tf.keras.layers.Dense(units, use_bias=False)
        output = activate(normalize(dense(input_layer)))
        return tf.keras.layers.Concatenate(name=name)([output, input_layer])

    return func
