import os
from shutil import copyfile
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed

from mathy.a3c.config import A3CArgs

from ..agent.layers.bahdanau_attention import BahdanauAttention
from ..agent.layers.lstm_stack import LSTMStack
from ..agent.layers.math_embedding import MathEmbedding
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from ..agent.layers.math_policy_resnet import MathPolicyResNet


class ActorCriticModel(tf.keras.Model):
    args: A3CArgs
    optimizer: tf.optimizers.Optimizer

    def __init__(
        self,
        args: A3CArgs,
        optimizer: tf.optimizers.Optimizer,
        predictions=2,
        initial_state: Any = None,
    ):
        super(ActorCriticModel, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.init_global_step()
        self.predictions = predictions
        # negative/neutral/positive
        self.reward_prediction = tf.keras.layers.Dense(3, name="rp_logits")
        self.rp_attention = BahdanauAttention(self.args.units, "rp_attn")
        self.pi_sequence = TimeDistributed(
            MathPolicyDropout(self.predictions), name="pi_head"
        )
        self.value_logits = tf.keras.layers.Dense(1, name="value_logits")
        self.embedding = MathEmbedding(self.args.units)

    def call(self, batch_features, apply_mask=True):
        inputs = batch_features
        # Extract features into contextual inputs, sequence inputs.
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)

        values = self.value_logits(context_inputs)
        logits = self.pi_sequence(sequence_inputs)
        trimmed_logits, mask_logits = self.apply_pi_mask(
            logits, batch_features, sequence_length
        )
        mask_result = trimmed_logits if not apply_mask else mask_logits
        return logits, values, mask_result

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
        negative_mask_logits = tf.where(
            tf.equal(mask_logits, tf.constant(0.0)),
            tf.fill(trim_logits.shape, -1000000.0),
            mask_logits,
        ).numpy()

        return trim_logits, negative_mask_logits

    def maybe_load(self, initial_state=None, do_init=False):
        if initial_state is not None:
            # NOTE: This is needed to properly initialize the trainable vars
            #       for the global model. Each prediction path must be called
            #       here or you'll get gradient descent errors about variables
            #       not matching gradients when the local_model tries to optimize
            #       against the global_model.
            self.call(initial_state)
            self.predict_reward(initial_state)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_path = os.path.join(self.args.model_dir, self.args.model_name)

        if do_init and self.args.init_model_from is not None:
            init_model_path = os.path.join(
                self.args.init_model_from, self.args.model_name
            )
            if os.path.exists(init_model_path) and not os.path.exists(model_path):
                print(f"initialize model weights from: {init_model_path}")
                copyfile(init_model_path, model_path)
            else:
                raise ValueError(f"could not initialize model from: {init_model_path}")

        if os.path.exists(model_path):
            if do_init:
                print("Loading model from: {}".format(model_path))
            if os.path.exists(model_path):
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

    def call_masked(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        logits, values, masked = self.call(inputs)
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits).numpy()
        return logits, values, probs

    def predict_one(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict one probability distribution and value for the
        given sequence of inputs"""
        logits, values, masked = self.call(inputs)
        # take the last timestep
        masked = masked[-1]
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits).numpy()
        return probs, tf.squeeze(values[-1]).numpy()

    def predict_reward(self, experience_frames) -> tf.Tensor:
        """Reward prediction for auxiliary tasks. Given an input sequence
        predict a class for positive, negative or neutral."""
        inputs = experience_frames
        # Extract features into contextual inputs, sequence inputs.
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)

        # seq_last = tf.expand_dims(tf.reshape(sequence_inputs[-1], [-1]), axis=0)
        ctx_one = tf.expand_dims(context_inputs[-1], axis=0)
        attn_out, attn_weights = self.rp_attention(sequence_inputs[-1], ctx_one)

        # attn_out, attn_weights = self.rp_attention(sequence_inputs, context_inputs)
        flat_attention = tf.reshape(attn_out, [1, -1])
        predict_logits = self.reward_prediction(flat_attention)
        values = tf.squeeze(predict_logits)
        return tf.nn.softmax(values)
