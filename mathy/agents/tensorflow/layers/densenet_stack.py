import tensorflow as tf
from .densenet_block import DenseNetBlock
from typing import List, Optional
from ..swish import swish


class DenseNetStack(tf.keras.layers.Layer):
    """DenseNet like stack of residually connected layers where the input and all
    of the previous outputs are provided as the input to each layer:

        https://arxiv.org/pdf/1608.06993.pdf

    From the paper: "Crucially, in contrast to ResNets, we never combine features
    through summation before they are passed into a layer; instead, we combine features
    by concatenating them"
    """

    def __init__(
        self,
        units=64,
        num_layers=4,
        random_seed=1337,
        layer_scaling_factor=0.75,
        share_weights=False,
        activation=swish,
        output_transform: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        self.units = units
        self.layer_scaling_factor = layer_scaling_factor
        self.num_layers = num_layers
        self.output_transform = output_transform
        self.activate = activation
        self.concat = tf.keras.layers.Concatenate()
        block_units = int(self.units)
        self.dense_stack = [
            DenseNetBlock(block_units, name="densenet_0", use_shared=share_weights)
        ]
        for i in range(self.num_layers - 1):
            block_units = int(block_units * self.layer_scaling_factor)
            self.dense_stack.append(
                DenseNetBlock(
                    block_units, name=f"densenet_{i + 1}", use_shared=share_weights
                )
            )
        super(DenseNetStack, self).__init__(**kwargs)

    def call(self, input_tensor: tf.Tensor):
        stack_inputs: List[tf.Tensor] = []
        root_input = input_tensor
        # Iterate the stack and call each layer, also inputting a list
        # of all the previous stack layers outputs.
        for layer in self.dense_stack:
            # Save reference to input
            prev_tensor = input_tensor
            # Apply densnet block
            input_tensor = layer(input_tensor, stack_inputs)
            # Append the current input to the list of previous input tensors
            stack_inputs.append(prev_tensor)
        output = self.concat([input_tensor, root_input])
        if self.output_transform is not None:
            output = self.output_transform(output)
        return self.activate(output) if self.activate is not None else output
