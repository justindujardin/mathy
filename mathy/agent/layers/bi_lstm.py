import tensorflow as tf


def BiLSTM(name="bi_lstm", time_major=False):
    """Bi-directional LSTM for processing input vectors, optionally seeded
    with state from another input. Usually this would be something like
    non-sequential features that are associated with your sequence features"""

    def func(input_layer, initial_state=None):
        units = input_layer.get_shape()[2]
        forward_layer = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            time_major=time_major,
            name=f"{name}/forward",
        )
        backward_layer = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            go_backwards=True,
            return_state=True,
            time_major=time_major,
            name=f"{name}/backward",
        )

        # Support seeding the LSTM with initial state from some other input
        # Inspired by: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
        if initial_state is not None:
            dense_h = tf.keras.layers.Dense(units, name=f"{name}/initial_h")(
                initial_state
            )
            dense_c = tf.keras.layers.Dense(units, name=f"{name}/initial_c")(
                initial_state
            )
            initial_state = [dense_h, dense_c]

        # TODO: make sequence masking work
        # input_layer = tf.keras.layers.Masking()(input_layer)

        lstm_fwd, state_h_fwd, state_c_fwd = forward_layer(
            input_layer, initial_state=initial_state
        )
        lstm_bwd, state_h_bwd, state_c_bwd = backward_layer(
            input_layer, initial_state=[state_h_fwd, state_c_fwd]
        )

        return (
            tf.keras.layers.Add(name=f"{name}/hidden_states")(
                [state_h_fwd, state_h_bwd]
            ),
            tf.keras.layers.Add(name=f"{name}/add")([lstm_fwd, lstm_bwd, input_layer]),
        )

    return func
