import tensorflow as tf
from .residual_dense import ResidualDense


class MathPolicy(tf.keras.layers.Layer):
    """Math policy predictor for sequence inputs that utilizes a residual tower for feature extraction"""

    def __init__(self, num_predictions=2, **kwargs):
        self.num_predictions = num_predictions
        self.activate = tf.keras.layers.Dense(num_predictions, activation="softmax")
        self.stack_height = 4
        self.dense_stack = [ResidualDense()]
        for i in range(self.stack_height - 1):
            self.dense_stack.append(ResidualDense())
        super(MathPolicy, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor):
        for layer in self.dense_stack:
            input_tensor = layer(input_tensor)
        return self.activate(input_tensor)

