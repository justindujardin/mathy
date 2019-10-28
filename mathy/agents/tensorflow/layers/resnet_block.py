import tensorflow as tf
from ..swish import swish


class ResNetBlock(tf.keras.layers.Layer):
    """Dense layer with residual input/output addition to help with backpropagation"""

    def __init__(self, units=128, layer_norm=True, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        if layer_norm:
            self.normalize = tf.keras.layers.LayerNormalization()
        else:
            self.normalize = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(
            units, use_bias=False, name="resdense", activation=swish
        )
        self.combine = tf.keras.layers.Add(name=f"{self.name}_add")

    def call(self, input_tensor):
        with tf.compat.v1.variable_scope(self.name):
            return self.combine(
                [self.normalize(self.dense(input_tensor)), input_tensor]
            )

