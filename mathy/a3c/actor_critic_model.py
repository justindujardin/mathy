from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from typing import Optional, Any
from ..agent.layers.math_embedding import MathEmbedding
from ..agent.layers.lstm_stack import LSTMStack
from ..agent.layers.resnet_stack import ResNetStack
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from ..agent.layers.bahdanau_attention import BahdanauAttention
from tensorflow.keras.layers import TimeDistributed
import os
from shutil import copyfile

from mathy.a3c.config import A3CArgs


class ActorCriticModel(tf.keras.Model):
    args: A3CArgs
    optimizer: tf.optimizers.Optimizer

    def __init__(
        self,
        args: A3CArgs,
        optimizer: tf.optimizers.Optimizer,
        predictions=2,
        shared_layers=None,
        initial_state: Any = None,
    ):
        super(ActorCriticModel, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.init_global_step()
        self.predictions = predictions
        self.shared_layers = shared_layers
        self.pi_logits = tf.keras.layers.Dense(predictions, name="pi_logits")
        self.resnet = ResNetStack(units=args.units, name="resnet", num_layers=4)
        self.pi_sequence = TimeDistributed(
            MathPolicyDropout(self.predictions), name="pi_head"
        )
        self.lstm = LSTMStack(units=args.units, share_weights=True)
        self.value_logits = tf.keras.layers.Dense(1, name="value_logits")
        self.embedding = MathEmbedding()
        self.attention = BahdanauAttention(self.args.units)

    def call(self, batch_features):
        inputs = batch_features
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                inputs = layer(inputs)
        # Extract features into contextual inputs, sequence inputs.
        context_inputs, lstm_vectors, hidden_states, sequence_length = self.embedding(
            inputs
        )

        feature_vectors = self.resnet(lstm_vectors)
        attention_context, attention_weights = self.attention(
            lstm_vectors, context_inputs
        )

        values = self.value_logits(attention_context)
        logits = self.pi_sequence(feature_vectors)
        masked_logits = self.apply_pi_mask(logits, batch_features, sequence_length)
        return logits, values, masked_logits

    def apply_pi_mask(self, logits, batch_features, sequence_length: int):
        """Take the policy_mask from a batch of features and multiply
        the policy logits by it to remove any invalid moves from
        selection """
        batch_mask_flat = tf.reshape(
            batch_features["policy_mask"], (logits.shape[0], -1, self.predictions)
        )
        # Trim the logits to match the feature mask
        trim_logits = logits[:, : batch_mask_flat.shape[1], :]
        features_mask = tf.cast(batch_mask_flat, dtype=tf.float32)
        mask_logits = tf.multiply(trim_logits, features_mask)
        mask_logits = tf.where(
            tf.equal(mask_logits, tf.constant(0.0)),
            tf.fill(trim_logits.shape, -1000000.0),
            mask_logits,
        ).numpy()

        return mask_logits

    def maybe_load(self, initial_state=None, do_init=False):
        if initial_state is not None:
            self.call(initial_state)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_path = os.path.join(self.args.model_dir, self.args.model_name)

        if do_init and self.args.init_model_from is not None:
            init_model_path = os.path.join(
                self.args.init_model_from, self.args.model_name
            )
            if os.path.exists(init_model_path):
                print(f"initialize model weights from: {init_model_path}")
                copyfile(init_model_path, model_path)
            else:
                raise ValueError(f"could not initialize model from: {init_model_path}")

        if os.path.exists(model_path):
            if do_init:
                print("Loading model from: {}".format(model_path))
            self.load_weights(model_path)

    def init_global_step(self):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        name = f"{self.args.model_name}.step"
        step_path = os.path.join(self.args.model_dir, name)
        step_num = 0
        if os.path.exists(step_path):
            with open(step_path, "r") as f:
                step_num = int(f.readline())
        self.global_step = tf.Variable(
            step_num, trainable=False, name="global_step", dtype=tf.int64
        )
        self.save_global_step(step_num)

    def save_global_step(self, step: int = 0):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        name = f"{self.args.model_name}.step"
        step_path = os.path.join(self.args.model_dir, name)
        with open(step_path, "w") as f:
            f.write(f"{step}")

    def save(self):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.save_weights(model_path)
        step = self.global_step.numpy()
        print(f"[save] step({step}) model({model_path})")
        self.save_global_step(step)

    def call_masked(self, inputs, mask) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        logits, values, masked = self.call(inputs)
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits).numpy()
        return logits, values, probs
