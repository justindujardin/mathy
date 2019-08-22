import tensorflow as tf
from .resnet_block import ResNetBlock


class MathPolicyResNet(tf.keras.layers.Layer):
    """Policy that passes inputs through a ResNet tower for feature extraction before
    applying Dropout and Softmax"""

    def __init__(
        self,
        num_predictions=2,
        dropout=0.1,
        random_seed=1337,
        num_layers=4,
        units=128,
        **kwargs
    ):
        self.num_predictions = num_predictions
        self.activate = tf.keras.layers.Dense(num_predictions, activation="softmax")
        self.dropout = tf.keras.layers.Dropout(dropout, seed=random_seed)
        self.stack_height = num_layers
        self.dense_stack = [ResNetBlock(units=units)]
        for i in range(self.stack_height - 1):
            self.dense_stack.append(ResNetBlock(units=units))
        super(MathPolicyResNet, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor):
        for layer in self.dense_stack:
            input_tensor = layer(input_tensor)
        # NOTE: Apply dropout AFTER ALL batch norms in the resnet:
        #       see: https://arxiv.org/pdf/1801.05134.pdf
        input_tensor = self.dropout(input_tensor)
        return self.activate(input_tensor)

