import tensorflow as tf
from .densenet_block import DenseNetBlock
from .bahdanau_attention import BahdanauAttention


class DenseNetStack(tf.keras.layers.Layer):
    """DenseNet like stack of residually connected layers where the input and all of the previous
    outputs are provided as the input to each layer: https://arxiv.org/pdf/1608.06993.pdf"""

    def __init__(
        self,
        units=64,
        num_layers=4,
        dropout=0.2,
        random_seed=1337,
        layer_scaling_factor=0.75,
        **kwargs
    ):
        self.units = units
        self.layer_scaling_factor = layer_scaling_factor
        self.num_layers = num_layers
        self.activate = tf.keras.layers.Activation("relu")
        self.dropout = tf.keras.layers.Dropout(dropout, seed=random_seed)
        block_units = int(self.units)
        self.dense_stack = [DenseNetBlock(block_units)]
        for i in range(self.num_layers - 1):
            block_units = int(block_units * self.layer_scaling_factor)
            self.dense_stack.append(DenseNetBlock(block_units))
        super(DenseNetStack, self).__init__(**kwargs)

    def call(self, input_tensor):
        stack_inputs = []
        # Iterate the stack and call each layer, also inputting a list
        # of all the previous stack layers outputs.
        for layer in self.dense_stack:
            # Save reference to input
            prev_tensor = input_tensor
            # Apply densnet block
            input_tensor = layer(input_tensor, stack_inputs)
            # Append the current input to the list of previous input tensors
            stack_inputs.append(prev_tensor)

        # NOTE: Apply dropout AFTER ALL batch norms in the resnet:
        #       see: https://arxiv.org/pdf/1801.05134.pdf
        input_tensor = self.dropout(input_tensor)
        return self.activate(input_tensor)

