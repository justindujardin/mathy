import tensorflow as tf


def ResNetBlock(name="residual_block"):
    """Dense layer with residual input addition for help with backpropagation"""

    def func(input_layer):
        activate = tf.keras.layers.Activation("relu")
        normalize = tf.keras.layers.BatchNormalization()
        dense = tf.keras.layers.Dense(input_layer.get_shape()[1], use_bias=False)
        output = normalize(activate(dense(input_layer)))
        return tf.keras.layers.Add(name=name)([output, input_layer])

    return func
