import tensorflow as tf


def ResNetBlock(name="residual_block", units=128, use_shared=False):
    """Dense layer with residual input/output addition to help with backpropagation"""

    def build():
        activate = tf.keras.layers.Activation("relu", name="relu")
        normalize = tf.keras.layers.BatchNormalization()
        dense = tf.keras.layers.Dense(units, use_bias=False, name="resdense")
        combine = tf.keras.layers.Add(name=f"{name}_add")
        return normalize, dense, activate, combine

    if use_shared:
        with tf.compat.v1.variable_scope(name, reuse=True, auxiliary_name_scope=False):
            normalize, dense, activate, combine = build()
    else:
        with tf.compat.v1.variable_scope(name, auxiliary_name_scope=False):
            normalize, dense, activate, combine = build()

    @tf.function
    def func(input_layer):
        with tf.compat.v1.variable_scope(name):
            return combine([normalize(activate(dense(input_layer))), input_layer])

    return func
