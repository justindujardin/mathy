import tensorflow as tf


class ResNetBlock(tf.keras.layers.Layer):
    """Dense layer with residual input/output addition to help with backpropagation"""

    def __init__(self, units=128, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.activate = tf.keras.layers.Activation("relu", name="relu")
        self.normalize = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(units, use_bias=False, name="resdense")
        self.combine = tf.keras.layers.Add(name=f"{self.name}_add")

    def call(self, input_tensor):
        with tf.compat.v1.variable_scope(self.name):
            return self.combine(
                [self.normalize(self.activate(self.dense(input_tensor))), input_tensor]
            )

