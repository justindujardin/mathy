import os
from shutil import copyfile
from typing import Any, Optional, Tuple
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed

from mathy.a3c.config import A3CArgs
from ..state import MathyWindowObservation, MathyWindowObservation
from ..agent.layers.bahdanau_attention import BahdanauAttention
from ..agent.layers.lstm_stack import LSTMStack
from .embedding import MathyEmbedding
from ..agent.layers.math_policy_dropout import MathPolicyDropout
from ..agent.layers.math_policy_resnet import MathPolicyResNet


class PolicySequences(tf.keras.layers.Layer):
    def __init__(self, num_predictions=2, dropout=0.1, random_seed=1337, **kwargs):
        super(PolicySequences, self).__init__(**kwargs)
        self.value_dense = tf.keras.layers.Dense(
            64, name="value_dense", kernel_initializer="he_uniform"
        )
        self.value_layer = tf.keras.layers.Dense(
            1, name="value_logits", kernel_initializer="he_uniform"
        )
        self.num_predictions = num_predictions
        self.logits = tf.keras.layers.Dense(
            num_predictions, name="pi_logits_dense", kernel_initializer="he_uniform"
        )
        self.dropout = tf.keras.layers.Dropout(dropout, seed=random_seed)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.num_predictions])

    def call(self, input_tensor: tf.Tensor):
        # NOTE: Apply dropout AFTER ALL batch norms. This works because
        # the MathPolicy is right before the final Softmax output. If more
        # layers with BatchNorm get inserted after this, the performance
        # could worsen. Inspired by: https://arxiv.org/pdf/1801.05134.pdf
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.logits(input_tensor)
        policy_out = tf.keras.activations.relu(input_tensor, alpha=0.0001)

        return policy_out


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
        self.value_logits = tf.keras.layers.Dense(1, name="policy_value/value_logits")
        self.policy_logits = TimeDistributed(
            PolicySequences(self.predictions), name="policy_value/policy_logits"
        )
        self.attention = BahdanauAttention(self.args.units, "policy_value/attention")

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
        self.vr_flatten = tf.keras.layers.Flatten(name="value_replay/flatten")
        self.vr_prepare_values = tf.keras.layers.Dense(
            self.args.embedding_units, name="value_replay/prepare_values"
        )

    def call(
        self,
        features_window: MathyWindowObservation,
        apply_mask=True,
        reset_states=False,
    ):
        # batch_size = len(features_window.nodes)
        # start = time.time()
        inputs = features_window
        # Extract features into contextual inputs, sequence inputs.
        sequence_inputs, sequence_length = self.embedding(
            inputs, reset_states=reset_states
        )
        values = self.value_logits(tf.reduce_mean(sequence_inputs, axis=1))
        logits = self.policy_logits(sequence_inputs)
        trimmed_logits, mask_logits = self.apply_pi_mask(
            logits, features_window, sequence_length
        )
        mask_result = trimmed_logits if not apply_mask else mask_logits
        # if batch_size > 1:
        #     print(
        #         "call took : {0:03f} for batch of {1:03}".format(
        #             time.time() - start, batch_size
        #         )
        #     )
        return logits, values, mask_result

    def apply_pi_mask(
        self, logits, features_window: MathyWindowObservation, sequence_length: int
    ):
        """Take the policy_mask from a batch of features and multiply
        the policy logits by it to remove any invalid moves from
        selection """

        mask = tf.convert_to_tensor(features_window.mask, dtype=tf.float32)
        features_mask = tf.reshape(mask, (logits.shape[0], -1, self.predictions))
        # Trim the logits to match the feature mask
        trim_logits = logits[:, : features_mask.shape[1], :]
        mask_logits = tf.multiply(trim_logits, features_mask)
        negative_mask_logits = tf.where(
            tf.equal(mask_logits, tf.constant(0.0)),
            tf.fill(trim_logits.shape, -1000000.0),
            mask_logits,
        ).numpy()

        return trim_logits, negative_mask_logits

    def call_masked(
        self, inputs: MathyWindowObservation
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        logits, values, masked = self.call(inputs)
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits).numpy()
        return logits, values, probs

    def predict_next(
        self, inputs: MathyWindowObservation
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
        sequence_inputs, sequence_length = self.embedding(inputs)

        combined_features = self.rp_prepare_values(
            tf.reduce_sum(sequence_inputs, axis=1)
        )
        predict_logits = self.reward_prediction(
            tf.expand_dims(combined_features[-1], axis=0)
        )
        # Output 3 class logits and convert to probabilities with softmax
        return tf.nn.softmax(tf.squeeze(predict_logits))

    def predict_value_replays(self, inputs: MathyWindowObservation) -> tf.Tensor:
        """Value replay aux task that replays historical observations
        and makes new value predictions using the features learned from
        the reward prediction task"""
        if self.args.use_value_replay is False:
            raise EnvironmentError(
                "This model is not configured to use value replay. "
                "To use the value replay aux task, set 'use_value_replay=True'"
            )

        # Pass the sequence inputs through the shared embedding network
        sequence_inputs, sequence_length = self.embedding(inputs)
        # vr_in = self.vr_flatten(sequence_inputs)
        # vr_in = self.vr_prepare_values(vr_in)
        # vr_in = tf.expand_dims(vr_in, axis=0)
        values = self.value_logits(sequence_inputs)
        return values

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
            self.embedding.call(initial_state)
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
