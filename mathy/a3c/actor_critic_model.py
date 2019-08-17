import tensorflow as tf
from ..agent.layers.math_embedding import MathEmbedding
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from tensorflow.keras.layers import TimeDistributed


class ActorCriticModel(tf.keras.Model):
    def __init__(self, units=128, predictions=2, shared_layers=None):
        super(ActorCriticModel, self).__init__()
        self.predictions = predictions
        self.shared_layers = shared_layers
        self.in_dense = tf.keras.layers.Dense(units)
        self.value_dense = tf.keras.layers.Dense(units)
        self.pi_logits = tf.keras.layers.Dense(predictions)
        self.pi_sequence = TimeDistributed(
            MathPolicyDropout(self.predictions, dropout=0.2), name="policy_head"
        )
        self.value_logits = tf.keras.layers.Dense(1)
        self.embedding = MathEmbedding()

    def call(self, inputs):

        # Extract features into a contextual inputs layer, and a sequence
        # inputs layer with the total sequence length.
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)
        inputs = self.in_dense(context_inputs)
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                inputs = layer(inputs)
        logits = self.pi_sequence(sequence_inputs)
        values = self.value_logits(self.value_dense(inputs))
        return logits, values

