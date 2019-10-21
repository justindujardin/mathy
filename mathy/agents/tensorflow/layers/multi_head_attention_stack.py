import tensorflow as tf

from .multi_head_attention import MultiHeadAttention
from .densenet_stack import DenseNetStack


class MultiHeadAttentionStack(tf.keras.layers.Layer):
    """Stack of Multi-Head Attention layers with a shared DenseNet stack
    between each layer"""

    def __init__(
        self, num_layers=2, num_heads=4, dropout_rate=0.1, attn_width=128, **kwargs
    ):
        super(MultiHeadAttentionStack, self).__init__(**kwargs)
        self.stack_height = num_layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.memory = DenseNetStack(
            output_transform=tf.keras.layers.Dense(units=attn_width)
        )
        self.stack = []
        for i in range(self.stack_height):
            self.stack.append(MultiHeadAttention(num_heads, name=f"{self.name}_{i}"))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], None])

    def call(self, input_tensor: tf.Tensor):
        layer = self.stack[0]
        stack = self.stack[1:]
        output = layer(input_tensor)
        for layer in stack:
            output = in_one = layer(output)
            output = self.dropout(output)
            output = in_two = self.norm(self.add([in_one, output]))
            output = self.memory(output)
            output = self.norm(self.add([in_two, output]))
        return output
