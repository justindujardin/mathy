import numpy as np
import tensorflow as tf
from typing import Optional, Any
from ..agent.layers.math_embedding import MathEmbedding
from ..agent.layers.lstm_stack import LSTMStack
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from tensorflow.keras.layers import TimeDistributed
import os


class ActorCriticModel(tf.keras.Model):
    def __init__(
        self,
        units=128,
        predictions=2,
        shared_layers=None,
        save_dir: str = "/tmp",
        load_model: Optional[str] = None,
        initial_state: Any = None,
    ):
        super(ActorCriticModel, self).__init__()
        self.save_dir = save_dir
        self.load_model = load_model
        self.predictions = predictions
        self.shared_layers = shared_layers
        self.in_dense = tf.keras.layers.Dense(units)
        self.value_dense = tf.keras.layers.Dense(units)
        self.pi_logits = tf.keras.layers.Dense(predictions)
        self.pi_sequence = TimeDistributed(
            MathPolicyDropout(self.predictions, dropout=0.2), name="policy_head"
        )
        self.lstm = LSTMStack(units=units, share_weights=True)
        self.value_logits = tf.keras.layers.Dense(1)
        self.embedding = MathEmbedding()

    def call(self, inputs):

        # Extract features into a contextual inputs layer, and a sequence
        # inputs layer with the total sequence length.
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)

        hidden_states, lstm_vectors = self.lstm(sequence_inputs, context_inputs)

        inputs = self.in_dense(hidden_states)
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                inputs = layer(inputs)
        logits = self.pi_sequence(lstm_vectors)
        values = self.value_logits(self.value_dense(inputs))
        return logits, values

    def maybe_load(self, initial_state=None):
        if initial_state is not None:
            self.call(initial_state)
        if self.load_model is not None:
            model_path = os.path.join(self.save_dir, f"{self.load_model}.h5")
            if os.path.exists(model_path):
                print("Loading model from: {}".format(model_path))
                self.load_weights(model_path)

    def call_masked(self, inputs, mask):
        logits, values = self.call(inputs)
        probs = tf.nn.softmax(tf.squeeze(logits))
        # Flatten for action selection and masking
        probs = tf.reshape(probs, [-1]).numpy()
        mask = mask[:]
        while len(mask) < len(probs):
            mask.append(0.0)
        probs *= mask
        pi_sum = np.sum(probs)
        if pi_sum > 0:
            probs /= pi_sum
        return logits, values, probs

