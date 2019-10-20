import tensorflow as tf

from .keras_self_attention.multi_head_attention import MultiHeadAttention


class MultiHeadAttentionStack(tf.keras.layers.Layer):
    """Stack of Multi-Head attention layers"""

    def __init__(self, num_layers=2, num_heads=4, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttentionStack, self).__init__(**kwargs)
        self.stack_height = num_layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
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
            attn_out = layer(output)
            attn_out = self.dropout(attn_out)
            output = self.norm(self.add([attn_out, output]))
        return output
