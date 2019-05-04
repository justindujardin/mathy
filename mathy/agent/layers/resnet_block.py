import tensorflow as tf


def ResNetBlock(name="residual_block", units=128, use_shared=False):
    """Dense layer with residual input addition for help with backpropagation"""

    def build():
        activate = tf.keras.layers.Activation("relu", name="relu")
        normalize = tf.keras.layers.BatchNormalization()
        dense = tf.keras.layers.Dense(units, use_bias=False, name="resdense")
        concat = tf.keras.layers.Concatenate(name=f"{name}_concat")
        return normalize, dense, activate, concat

    if use_shared:
        with tf.compat.v1.variable_scope(name, reuse=True, auxiliary_name_scope=False):
            normalize, dense, activate, concat = build()
    else:
        with tf.compat.v1.variable_scope(name, auxiliary_name_scope=False):
            normalize, dense, activate, concat = build()

    def func(input_layer):
        with tf.compat.v1.variable_scope(name):
            return concat([normalize(activate(dense(input_layer))), input_layer])

    return func
