import tensorflow as tf
from .resnet_block import ResNetBlock


class ResNetStack(tf.keras.layers.Layer):
    """Stack of ResNet layers"""

    def __init__(self, units=64, num_layers=2, share_weights=False, **kwargs):
        self.stack_height = num_layers
        self.dense_stack = [
            ResNetBlock(name=f"res_block_0", units=units, use_shared=share_weights)
        ]
        for i in range(self.stack_height - 1):
            self.dense_stack.append(
                ResNetBlock(
                    name=f"res_block_{i + 1}", units=units, use_shared=share_weights
                )
            )
        super(ResNetStack, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor):
        with tf.compat.v1.variable_scope(self.name):
            for layer in self.dense_stack:
                input_tensor = layer(input_tensor)
            return input_tensor

