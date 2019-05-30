import tensorflow as tf


class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_size, action_size, shared_layers=None):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.shared_layers = shared_layers
        self.pi_dense = tf.keras.layers.Dense(128)
        self.value_dense = tf.keras.layers.Dense(128)
        self.pi_logits = tf.keras.layers.Dense(action_size)
        self.value_logits = tf.keras.layers.Dense(1)

    def call(self, inputs):
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                inputs = layer(inputs)
        logits = self.pi_logits(self.pi_dense(inputs))
        values = self.value_logits(self.value_dense(inputs))
        return logits, values

