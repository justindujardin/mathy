import tensorflow as tf


def DenseNetBlock(units=256, name="densenet_block"):
    """DenseNet like block layer that concatenates inputs with extracted features 
    and feeds them into all future tower layers."""

    def func(input_layer, previous_layers):
        with tf.compat.v1.variable_scope(name):
            inputs = previous_layers[:]
            inputs.append(input_layer)
            # concatenate the input and previous layer inputs
            if (len(inputs)) > 1:
                input_layer = tf.keras.layers.Concatenate()(inputs)
            activate = tf.keras.layers.Activation("relu")
            normalize = tf.keras.layers.BatchNormalization()
            dense = tf.keras.layers.Dense(units, use_bias=False)
            output = normalize(activate(dense(input_layer)))
            return output

    return func
