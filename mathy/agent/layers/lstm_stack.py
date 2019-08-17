import tensorflow as tf

from .lstm import LSTM


class LSTMStack(tf.keras.layers.Layer):
    """Stack of LSTM layers"""

    def __init__(self, units=64, num_layers=2, share_weights=False, **kwargs):
        self.stack_height = num_layers
        self.stack = [LSTM(name=f"lstm_0", units=units, use_shared=share_weights)]
        self.dense_h = tf.keras.layers.Dense(units, name=f"initial_h")
        self.dense_c = tf.keras.layers.Dense(units, name=f"initial_c")
        for i in range(self.stack_height - 1):
            self.stack.append(
                LSTM(name=f"lstm_{i + 1}", units=units, use_shared=share_weights)
            )
        super(LSTMStack, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    @tf.function
    def call(self, input_tensor, initial_states):
        # Prepare initial state as transformations from contextual features
        hidden_states = [self.dense_h(initial_states), self.dense_c(initial_states)]
        for layer in self.stack:
            hidden_states, input_tensor = layer(input_tensor, hidden_states)
        # Drop the cell state at the end
        hidden_states = hidden_states[0]
        return hidden_states, input_tensor
