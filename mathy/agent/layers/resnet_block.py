import tensorflow as tf


def ResNetBlock(name="residual_block"):
    """Dense layer with residual input addition for help with backpropagation"""

    def func(input_layer):
        with tf.compat.v1.variable_scope(name):
            activate = tf.keras.layers.Activation("relu", name="relu")
            normalize = tf.keras.layers.BatchNormalization()
            dense = tf.keras.layers.Dense(
                input_layer.get_shape()[1], use_bias=False, name="dense"
            )
            output = normalize(activate(dense(input_layer)))
            return tf.keras.layers.Add(name=f"{name}_add")([output, input_layer])

    return func
