import os
import pickle
import time
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import srsly
import tensorflow as tf
from tensorflow.keras import backend as K
from wasabi import msg

from ..env import MathyEnv
from ..envs import PolySimplify
from ..state import (
    MathyInputsType,
    MathyObservation,
    MathyWindowObservation,
    ObservationFeatureIndices,
    observations_to_window,
)
from ..util import print_error
from .base_config import BaseConfig
from .embedding import MathyEmbedding


class PolicyValueModel(tf.keras.Model):
    args: BaseConfig
    optimizer: tf.optimizers.Optimizer

    def __init__(
        self,
        args: BaseConfig = None,
        predictions: int = 2,
        initial_state: Any = None,
        **kwargs,
    ):
        super(PolicyValueModel, self).__init__(**kwargs)
        if args is None:
            args = BaseConfig()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr_initial,
            decay_steps=args.lr_decay_steps,
            decay_rate=args.lr_decay_rate,
            staircase=args.lr_decay_staircase,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.args = args
        self.predictions = predictions
        self.embedding = MathyEmbedding(self.args)
        self.policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(
                        self.predictions,
                        name="policy_ts_hidden",
                        kernel_initializer="he_normal",
                        activation=None,
                    ),
                    name="policy_logits",
                ),
                tf.keras.layers.LayerNormalization(name="policy_layer_norm"),
            ],
            name="policy_head",
        )
        self.value_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.args.units,
                    name="value_hidden",
                    kernel_initializer="he_normal",
                    activation="relu",
                ),
                tf.keras.layers.LayerNormalization(name="value_layer_norm"),
                tf.keras.layers.Dense(
                    1,
                    name="value_logits",
                    kernel_initializer="he_normal",
                    activation=None,
                ),
            ],
            name="value_head",
        )
        self.reward_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.args.units,
                    name="reward_hidden",
                    kernel_initializer="he_normal",
                    activation="relu",
                ),
                tf.keras.layers.LayerNormalization(name="reward_layer_norm"),
                tf.keras.layers.Dense(
                    1,
                    name="reward_logits",
                    kernel_initializer="he_normal",
                    activation=None,
                ),
            ],
            name="reward_head",
        )
        self.loss = tf.Variable(
            0.0, trainable=False, name="loss_placeholder", dtype=tf.float32
        )

    def compute_output_shape(
        self, input_shapes: List[tf.TensorShape]
    ) -> tf.TensorShape:
        nodes_shape: tf.TensorShape = input_shapes[0]
        return tf.TensorShape(
            (
                nodes_shape.dims[0].value,
                nodes_shape.dims[1].value,
                self.args.embedding_units,
            )
        )

    def call(
        self, features_window: MathyInputsType
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        call_print = self.args.print_model_call_times
        nodes = features_window[ObservationFeatureIndices.nodes]
        batch_size = (
            len(nodes)
            if not isinstance(nodes, (tf.Tensor, np.ndarray))
            else nodes.shape[0]
        )
        start = time.time()
        inputs = features_window
        # Extract features into contextual inputs, sequence inputs.
        sequence_inputs, attention_weights = self.embedding(inputs)
        sequence_mean = tf.reduce_mean(sequence_inputs, axis=1)
        values = self.value_net(sequence_mean)
        reward_logits = self.reward_net(sequence_mean)
        logits = self.policy_net(sequence_inputs)
        mask_logits = self.apply_pi_mask(logits, features_window)
        if call_print is True:
            print(
                "call took : {0:03f} for batch {1}".format(
                    time.time() - start, batch_size
                )
            )
        return logits, values, mask_logits, reward_logits, attention_weights

    def apply_pi_mask(
        self, logits: tf.Tensor, features_window: MathyInputsType,
    ) -> tf.Tensor:
        """Take the policy_mask from a batch of features and multiply
        the policy logits by it to remove any invalid moves"""
        mask = features_window[ObservationFeatureIndices.mask]
        logits_shape = tf.shape(logits)
        features_mask = tf.reshape(
            mask, (logits_shape[0], -1, self.predictions), name="pi_mask_reshape"
        )
        features_mask = tf.cast(features_mask, dtype=tf.float32)
        mask_logits = tf.multiply(logits, features_mask, name="mask_logits")
        negative_mask_logits = tf.where(
            tf.equal(mask_logits, tf.constant(0.0)),
            tf.fill(tf.shape(logits), -1000000.0),
            mask_logits,
            name="softmax_negative_logits",
        )
        return negative_mask_logits

    def predict_next(self, inputs: MathyInputsType) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict one probability distribution and value for the
        given sequence of inputs """
        logits, values, masked, rewards, attention = self.call(inputs)
        # take the last timestep
        masked = masked[-1][:]
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits)
        return probs, tf.squeeze(values[-1])

    def save(self) -> None:
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        with open(f"{model_path}.optimizer", "wb") as f:
            pickle.dump(self.optimizer.get_weights(), f)
        model_path += ".h5"
        self.save_weights(model_path, save_format="keras")
        step = self.optimizer.iterations.numpy()
        print(f"[save] step({step}) model({model_path})")


def _load_model(
    model: PolicyValueModel, model_file: str, optimizer_file: str,
) -> PolicyValueModel:
    model.load_weights(model_file)
    model._make_train_function()
    with open(optimizer_file, "rb") as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)
    return model


def get_or_create_policy_model(
    args: BaseConfig,
    predictions: int,
    is_main=False,
    required=False,
    env: MathyEnv = None,
) -> PolicyValueModel:
    if env is None:
        env = PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    initial_state: MathyWindowObservation = observations_to_window([observation])

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, args.model_name)
    # Transfer weights/optimizer from a different model
    if is_main and args.init_model_from is not None:
        init_model_path = os.path.join(args.init_model_from, args.model_name)
        opt = f"{init_model_path}.optimizer"
        mod = f"{init_model_path}.h5"
        cfg = f"{init_model_path}.config.json"
        if os.path.exists(f"{model_path}.bytes"):
            print_error(
                ValueError("Model Exists"),
                f"Cannot initialize on top of model: {model_path}",
                print_error=False,
            )
        if os.path.exists(opt) and os.path.exists(mod) and os.path.exists(cfg):
            print(f"initialize model from: {init_model_path}")
            copyfile(opt, f"{model_path}.optimizer")
            copyfile(mod, f"{model_path}.h5")
            copyfile(cfg, f"{model_path}.config.json")
        else:
            print_error(
                ValueError("Model Exists"),
                f"Cannot initialize on top of model: {model_path}",
                print_error=False,
            )

    model = PolicyValueModel(args=args, predictions=predictions, name="agent")
    init_inputs = initial_state.to_inputs()
    model.compile(
        optimizer=model.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.build(initial_state.to_input_shapes())
    model.predict(init_inputs)
    model.predict_next(init_inputs)

    opt = f"{model_path}.optimizer"
    mod = f"{model_path}.h5"
    if os.path.exists(mod):
        if is_main and args.verbose:
            with msg.loading(f"Loading model: {mod}..."):
                _load_model(model, mod, opt)
            msg.good(f"Loaded model: {mod}")
        else:
            _load_model(model, mod, opt)

        # If we're doing transfer, reset optimizer steps
        if is_main and args.init_model_from is not None:
            msg.info("reset optimizer steps to 0 for transfer model")
            model.optimizer.iterations.assign(0)
    elif required:
        print_error(
            ValueError("Model Not Found"),
            f"Cannot find model: {mod}",
            print_error=False,
        )
    elif is_main:
        cfg = f"{model_path}.config.json"
        if args.verbose:
            msg.info(f"wrote model config: {cfg}")
        srsly.write_json(cfg, args.dict(exclude_defaults=False))

    return model


def load_policy_value_model(
    model_data_folder: str, silent: bool = False
) -> Tuple[PolicyValueModel, BaseConfig]:
    meta_file = Path(model_data_folder) / "model.config.json"
    if not meta_file.exists():
        raise ValueError(f"model meta not found: {meta_file}")
    args = BaseConfig(**srsly.read_json(str(meta_file)))
    model_file = Path(model_data_folder) / "model.h5"
    optimizer_file = Path(model_data_folder) / "model.optimizer"
    if not model_file.exists():
        raise ValueError(f"model not found: {model_file}")
    if not optimizer_file.exists():
        raise ValueError(f"optimizer not found: {optimizer_file}")
    env: MathyEnv = PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    initial_state: MathyWindowObservation = observations_to_window([observation])
    model = PolicyValueModel(args=args, predictions=env.action_size, name="agent")
    init_inputs = initial_state.to_inputs()
    model.compile(
        optimizer=model.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.build(initial_state.to_input_shapes())
    model.predict(init_inputs)
    model.predict_next(init_inputs)

    if not silent:
        with msg.loading(f"Loading model: {model_file}..."):
            _load_model(model, str(model_file), str(optimizer_file))
        msg.good(f"Loaded model: {model_file}")
    else:
        _load_model(model, str(model_file), str(optimizer_file))
    return model, args
