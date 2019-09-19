import os
from shutil import copyfile
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed

from .config import MathyArgs
from .embedding import MathyEmbedding

from ..agent.layers.bahdanau_attention import BahdanauAttention
from ..agent.layers.lstm_stack import LSTMStack
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from ..agent.layers.math_policy_resnet import MathPolicyResNet


class MathyModel(tf.keras.Model):
    args: MathyArgs

    def __init__(self, args: MathyArgs, predictions=2, initial_state: Any = None):
        super(MathyModel, self).__init__()
        self.args = args
        self.init_global_step()
        self.predictions = predictions
        self.build_shared_network()
        self.build_policy_value()
        self.build_reward_prediction()
        self.build_value_replay()

    def build_shared_network(self):
        """The shared embedding network for all tasks"""
        self.embedding = MathyEmbedding(
            self.args.embedding_units, name="shared_network"
        )

    def build_policy_value(self):
        """A3C policy/value network"""
        self.pi_sequence = TimeDistributed(
            MathPolicyDropout(self.predictions), name="a3c/policy_logits"
        )
        self.attention = BahdanauAttention(self.args.units, "a3c/attention")
        self.value_logits = tf.keras.layers.Dense(1, name="a3c/value_logits")

    def build_reward_prediction(self):
        if self.args.use_reward_prediction is False:
            return
        self.reward_prediction = tf.keras.layers.Dense(
            3, name="reward_prediction/class_logits"
        )
        self.rp_prepare_values = tf.keras.layers.Dense(
            self.args.embedding_units, name="reward_prediction/prepare_inputs"
        )

    def build_value_replay(self):
        if self.args.use_value_replay is False:
            return
        self.vr_prepare_values = tf.keras.layers.Dense(
            self.args.embedding_units, name="value_replay/prepare_values"
        )

    def call(self, batch_features, apply_mask=True, initial_state=None):
        inputs = batch_features
        # Extract features into contextual inputs, sequence inputs.
        context_inputs, sequence_inputs, sequence_length = self.embedding(
            inputs, initial_state=initial_state
        )
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
        batch_mask_flat = batch_mask_flat[:, : logits.shape[1], :]
        trim_logits = logits[:, : logits.shape[1], :]
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
            if self.args.use_reward_prediction:
                self.predict_next_reward(initial_state)
            if self.args.use_value_replay:
                self.predict_value_replays(initial_state)
            self.embedding.call(initial_state, initialize=True)
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
            if do_init and self.args.verbose:
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

    def predict_next(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict one probability distribution and value for the
        given sequence of inputs"""
        logits, values, masked = self.call(inputs)
        # take the last timestep
        masked = masked[-1]
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits).numpy()
        return probs, tf.squeeze(values[-1]).numpy()

    def predict_next_reward(self, inputs) -> tf.Tensor:
        """Reward prediction for auxiliary tasks. Given an input sequence of
        observations predict a class for positive, negative or neutral representing
        the amount of reward that is received at the next state. """
        if self.args.use_reward_prediction is False:
            raise EnvironmentError(
                "This model is not configured to use reward prediction. "
                "To use reward prediction aux task, set 'use_reward_prediction=True'"
            )

        # Pass the sequence inputs through the shared embedding network
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)

        combined_features = self.rp_prepare_values(
            tf.reduce_sum(sequence_inputs, axis=1)
        )
        predict_logits = self.reward_prediction(
            tf.expand_dims(combined_features[-1], axis=0)
        )
        # Output 3 class logits and convert to probabilities with softmax
        return tf.nn.softmax(tf.squeeze(predict_logits))

    def predict_value_replays(self, inputs) -> tf.Tensor:
        """Value replay aux task that replays historical observations
        and makes new value predictions using the features learned from
        the reward prediction task"""
        if self.args.use_value_replay is False:
            raise EnvironmentError(
                "This model is not configured to use value replay. "
                "To use the value replay aux task, set 'use_value_replay=True'"
            )

        # Pass the sequence inputs through the shared embedding network
        context_inputs, sequence_inputs, sequence_length = self.embedding(inputs)
        vr_in = self.vr_prepare_values(tf.reduce_mean(sequence_inputs, axis=1))
        vr_in = tf.expand_dims(vr_in, axis=0)
        values = self.value_logits(vr_in)
        return values
