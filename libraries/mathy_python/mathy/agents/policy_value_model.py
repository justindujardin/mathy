import os
import pickle
import time
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import srsly
import tensorflow as tf
from tensorflow.python.keras import backend as K
from wasabi import msg
from ..traceback import print_error

from ..state import MathyInputsType, MathyWindowObservation, ObservationFeatureIndices
from .base_config import BaseConfig
from .embedding import MathyEmbedding


class PolicyValueModel(tf.keras.Model):
    args: BaseConfig
    optimizer: tf.optimizers.Optimizer

    def __init__(
        self,
        args: Optional[BaseConfig] = None,
        predictions=2,
        initial_state: Any = None,
        **kwargs,
    ):
        if args is None:
            args = BaseConfig()
        super(PolicyValueModel, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(lr=args.lr)
        self.args = args
        self.predictions = predictions
        self.embedding = MathyEmbedding(
            self.args.units, self.args.lstm_units, self.args.embedding_units
        )
        self.value_logits = tf.keras.layers.Dense(
            1, name="value_logits", kernel_initializer="he_normal", activation=None,
        )
        self.policy_td_dense = tf.keras.layers.Dense(
            self.predictions,
            name="timestep_dense",
            kernel_initializer="he_normal",
            activation=None,
        )
        self.policy_logits = tf.keras.layers.TimeDistributed(
            self.policy_td_dense, name="policy_logits"
        )
        self.normalize_pi = tf.keras.layers.LayerNormalization(name="policy_layer_norm")
        self.normalize_v = tf.keras.layers.LayerNormalization(name="value_layer_norm")
        self.loss = tf.Variable(
            0.0, trainable=False, name="loss_placeholder", dtype=tf.float32
        )

    def call(self, features_window: MathyInputsType, apply_mask=True):
        call_print = False
        nodes = features_window[ObservationFeatureIndices.nodes]
        batch_size = (
            len(nodes)
            if not isinstance(nodes, (tf.Tensor, np.ndarray))
            else nodes.shape
        )
        start = time.time()
        inputs = features_window
        # Extract features into contextual inputs, sequence inputs.
        sequence_inputs = self.embedding(inputs)
        values = self.normalize_v(self.value_logits(self.embedding.state_h))
        logits = self.normalize_pi(self.policy_logits(sequence_inputs))
        mask_logits = self.apply_pi_mask(logits, features_window)
        mask_result = logits if not apply_mask else mask_logits
        if call_print is True:
            print(
                "call took : {0:03f} for batch of {1:03}".format(
                    time.time() - start, batch_size
                )
            )
        return logits, values, mask_result

    def apply_pi_mask(
        self, logits: tf.Tensor, features_window: MathyInputsType,
    ):
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

    @tf.function
    def call_graph(
        self, inputs: MathyInputsType
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Autograph optimized function"""
        return self.call(inputs)

    def predict_next(self, inputs: MathyInputsType) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict one probability distribution and value for the
        given sequence of inputs"""
        logits, values, masked = self.call(inputs)
        # take the last timestep
        masked = masked[-1][:]
        flat_logits = tf.reshape(tf.squeeze(masked), [-1])
        probs = tf.nn.softmax(flat_logits)
        return probs, tf.squeeze(values[-1])

    def save(self):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        if self.args.model_format == "keras":
            with open(f"{model_path}.optimizer", "wb") as f:
                pickle.dump(self.optimizer.get_weights(), f)
            model_path += ".h5"
        self.save_weights(model_path, save_format=self.args.model_format)
        step = self.optimizer.iterations.numpy()
        print(f"[save] step({step}) model({model_path})")


def _load_model(model: PolicyValueModel, model_file: str, optimizer_file: str):
    model.load_weights(model_file)
    model._make_train_function()
    with open(optimizer_file, "rb") as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)


def get_or_create_policy_model(
    args: BaseConfig,
    env_actions: int,
    initial_state: MathyWindowObservation,
    is_main=False,
    required=False,
) -> PolicyValueModel:

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, args.model_name)
    # Transfer weights/optimizer from a different model
    if is_main and args.init_model_from is not None:
        if args.model_format == "keras":
            init_model_path = os.path.join(args.init_model_from, args.model_name)
            opt = f"{init_model_path}.optimizer"
            mod = f"{init_model_path}.h5"
            if os.path.exists(f"{model_path}.h5"):
                print_error(
                    ValueError("Model Exists"),
                    f"Cannot initialize on top of model: {model_path}",
                    print_error=False,
                )
            if os.path.exists(opt) and os.path.exists(mod):
                print(f"initialize model from: {init_model_path}")
                copyfile(opt, f"{model_path}.optimizer")
                copyfile(mod, f"{model_path}.h5")
            else:
                print_error(
                    ValueError("Model Exists"),
                    f"Cannot initialize on top of model: {model_path}",
                    print_error=False,
                )
        elif args.model_format == "tf":
            raise ValueError(
                "TODO: copy file checkpoint, data, etc, but not tensorboard"
            )
            init_model_path = os.path.join(args.init_model_from, args.model_name)
            if not os.path.exists(model_path):
                if os.path.exists(init_model_path):
                    print(f"initialize model weights from: {init_model_path}")
                    copyfile(init_model_path, model_path)
            else:
                raise ValueError(
                    f"model already exists at: {model_path}, cannot initialize"
                )

    model = PolicyValueModel(args=args, predictions=env_actions, name="agent")
    model.compile(optimizer=model.optimizer, loss="binary_crossentropy")
    model.predict(initial_state.to_inputs())
    model.build(initial_state.to_input_shapes())

    if args.model_format == "keras":
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

    elif args.model_format == "tf":
        if os.path.exists(os.path.join(args.model_dir, "checkpoint")):
            if args.verbose:
                print("Loading model from: {}".format(model_path))
            model.load_weights(model_path)
        elif is_main:
            cfg = f"{model_path}.config.json"
            if args.verbose:
                msg.info(f"wrote model config: {cfg}")
            srsly.write_json(cfg, args.dict(exclude_defaults=False))
    else:
        print_error(
            ValueError("Unknown Format"),
            f"The model format ({args.model_format}) is not understood."
            f"Choose one of 'tf' or 'keras'.",
            print_error=False,
        )

    return model
