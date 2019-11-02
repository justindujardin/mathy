import tensorflow as tf
from typing import List
from ..swish import swish


class DenseNetBlock(tf.keras.layers.Layer):
    """DenseNet like block layer that concatenates inputs with extracted features
    and feeds them into all future tower layers."""

    def __init__(self, units=128, num_layers=2, use_shared=False, **kwargs):
        super(DenseNetBlock, self).__init__(**kwargs)
        self.use_shared = use_shared
        self.normalize = tf.keras.layers.LayerNormalization(
            name=f"{self.name}_layer_norm"
        )
        self.dense = tf.keras.layers.Dense(
            units,
            name=f"{self.name}_denseblock",
            activation=swish,
            kernel_initializer="he_normal",
        )
        self.concat = tf.keras.layers.Concatenate(name=f"{self.name}_concat")

    def do_op(self, input_tensor: tf.Tensor, previous_tensors: List[tf.Tensor]):
        inputs = previous_tensors[:]
        inputs.append(input_tensor)
        # concatenate the input and previous layer inputs
        if (len(inputs)) > 1:
            input_tensor = self.concat(inputs)
        return self.normalize(self.dense(input_tensor))

    def call(self, input_tensor: tf.Tensor, previous_tensors: List[tf.Tensor]):
        if self.use_shared:
            with tf.compat.v1.variable_scope(
                self.name, reuse=True, auxiliary_name_scope=False
            ):
                return self.do_op(input_tensor, previous_tensors)
        else:
            with tf.compat.v1.variable_scope(self.name):
                return self.do_op(input_tensor, previous_tensors)
