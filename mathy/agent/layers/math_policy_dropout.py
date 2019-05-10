import tensorflow as tf


class MathPolicyDropout(tf.keras.layers.Layer):
    """policy predictor that applies Dropout and activation"""

    def __init__(
        self,
        num_predictions=2,
        dropout=0.1,
        random_seed=1337,
        feature_layer=None,
        **kwargs,
    ):
        self.num_predictions = num_predictions
        self.feature_layer = feature_layer
        self.activate = tf.keras.layers.Dense(
            num_predictions, name="relu", activation="relu"
        )
        self.dropout = tf.keras.layers.Dropout(dropout, seed=random_seed)
        super(MathPolicyDropout, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor):
        if self.feature_layer is not None:
            input_tensor = self.feature_layer(input_tensor)
        # NOTE: Apply dropout AFTER ALL batch norms. This works because
        # the MathPolicy is right before the final Softmax output. If more
        # layers with BatchNorm get inserted after this, the performance
        # could worsen. Inspired by: https://arxiv.org/pdf/1801.05134.pdf
        input_tensor = self.dropout(input_tensor)
        return self.activate(input_tensor)

