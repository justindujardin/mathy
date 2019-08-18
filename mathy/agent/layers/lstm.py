import tensorflow as tf


def LSTM(units=32, name="lstm", use_shared=False, use_mask=False):
    """LSTM for processing input vectors, optionally seeded with state from another 
    input. Usually this would be something like non-sequential features that are 
    associated with your sequence features"""

    def build():
        lstm = tf.keras.layers.LSTM(
            units, return_sequences=True, return_state=True, name="lstm"
        )
        mask = tf.keras.layers.Masking()
        combine = tf.keras.layers.Add(name="add")
        return lstm, combine, mask

    if use_shared:
        with tf.compat.v1.variable_scope(name, reuse=True, auxiliary_name_scope=False):
            lstm, combine, mask = build()
    else:
        with tf.compat.v1.variable_scope(name, auxiliary_name_scope=False):
            lstm, combine, mask = build()

    def func(input_layer, initial_state=None):
        with tf.compat.v1.variable_scope(name):
            if use_mask is True:
                input_layer = mask(input_layer)
            # Support seeding the LSTM with initial state from some other input
            # Inspired by: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
            lstm_fwd, state_h_fwd, state_c_fwd = lstm(
                input_layer, initial_state=initial_state
            )
            return ([state_h_fwd, state_c_fwd], combine([lstm_fwd, input_layer]))

    return func
