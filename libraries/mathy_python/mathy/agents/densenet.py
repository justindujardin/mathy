from typing import List, Optional

import tensorflow as tf


class DenseNetBlock(tf.keras.layers.Layer):
    """DenseNet like block layer that concatenates inputs with extracted features
    and feeds them into all future tower layers."""

    def __init__(
        self,
        units=128,
        num_layers=2,
        use_shared=False,
        activation="relu",
        normalization="batch",
        **kwargs,
    ):
        super(DenseNetBlock, self).__init__(**kwargs)
        self.units = units
        self.use_shared = use_shared
        self.activate = tf.keras.layers.Activation("relu", name="relu")
        if normalization == "batch":
            self.normalize = tf.keras.layers.BatchNormalization()
        elif normalization == "layer":
            self.normalize = tf.keras.layers.LayerNormalization()
        else:
            raise ValueError(f"unkknown layer normalization style: {normalization}")
        self.dense = tf.keras.layers.Dense(
            units, use_bias=False, name=f"{self.name}_dense"
        )
        self.concat = tf.keras.layers.Concatenate(name=f"{self.name}_concat")
        self.activate = tf.keras.layers.Activation(activation, name="activate")

    def do_op(self, input_tensor: tf.Tensor, previous_tensors: List[tf.Tensor]):
        inputs = previous_tensors[:]
        inputs.append(input_tensor)
        # concatenate the input and previous layer inputs
        if (len(inputs)) > 1:
            input_tensor = self.concat(inputs)
        return self.normalize(self.activate(self.dense(input_tensor)))

    def call(self, input_tensor: tf.Tensor, previous_tensors: List[tf.Tensor]):
        if self.use_shared:
            with tf.compat.v1.variable_scope(
                self.name, reuse=True, auxiliary_name_scope=False
            ):
                return self.do_op(input_tensor, previous_tensors)
        else:
            with tf.compat.v1.variable_scope(self.name):
                return self.do_op(input_tensor, previous_tensors)


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
        layer_scaling_factor=0.75,
        share_weights=False,
        activation="relu",
        output_transform: Optional[tf.keras.layers.Layer] = None,
        normalization_style: str = "layer",
        **kwargs,
    ):
        self.units = units
        self.layer_scaling_factor = layer_scaling_factor
        self.num_layers = num_layers
        self.output_transform = output_transform
        if activation is not None:
            self.activate = tf.keras.layers.Activation(activation)
        else:
            self.activate = None
        self.concat = tf.keras.layers.Concatenate()
        block_units = int(self.units)
        self.dense_stack = [
            DenseNetBlock(
                block_units,
                name="densenet_0",
                use_shared=share_weights,
                normalization=normalization_style,
            )
        ]
        for i in range(self.num_layers - 1):
            block_units = int(block_units * self.layer_scaling_factor)
            self.dense_stack.append(
                DenseNetBlock(
                    block_units,
                    name=f"densenet_{i + 1}",
                    use_shared=share_weights,
                    normalization=normalization_style,
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
