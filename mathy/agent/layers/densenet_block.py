import tensorflow as tf


def DenseNetBlock(units=256, name="densenet_block", use_shared=False):
    """DenseNet like block layer that concatenates inputs with extracted features 
    and feeds them into all future tower layers."""

    def build():
        activate = tf.keras.layers.Activation("relu", name="relu")
        normalize = tf.keras.layers.BatchNormalization()
        dense = tf.keras.layers.Dense(units, use_bias=False, name=f"{name}_dense")
        concat = tf.keras.layers.Concatenate(name=f"{name}_concat")
        return normalize, dense, activate, concat

    if use_shared:
        with tf.compat.v1.variable_scope(name, reuse=True, auxiliary_name_scope=False):
            normalize, dense, activate, concat = build()
    else:
        with tf.compat.v1.variable_scope(name):
            normalize, dense, activate, concat = build()

    def func(input_layer, previous_layers):
        with tf.compat.v1.variable_scope(name):
            inputs = previous_layers[:]
            inputs.append(input_layer)
            # concatenate the input and previous layer inputs
            if (len(inputs)) > 1:
                input_layer = concat(inputs)
            return normalize(activate(dense(input_layer)))

    return func
