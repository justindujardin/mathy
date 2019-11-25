import pickle
import os
import time
from shutil import copyfile
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from ..state import MathyWindowObservation, MathyInputsType, MathyInputs
from .base_config import BaseConfig
from .embedding import MathyEmbedding
from .swish_activation import swish


class PolicyValueModel(tf.keras.Model):
    args: BaseConfig
    optimizer: tf.optimizers.Optimizer

    def __init__(
        self, args: BaseConfig, predictions=2, initial_state: Any = None, **kwargs,
    ):
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
        self.loss = tf.Variable(
            0.0, trainable=False, name="loss_placeholder", dtype=tf.float32
        )

    def call(self, features_window: MathyInputsType, apply_mask=True):
        call_print = False
        nodes = features_window[MathyInputs.nodes]
        batch_size = (
            len(nodes)
            if not isinstance(nodes, (tf.Tensor, np.ndarray))
            else nodes.shape
        )
        start = time.time()
        inputs = features_window
        # Extract features into contextual inputs, sequence inputs.
        sequence_inputs = self.embedding(inputs)
        values = tf.reduce_mean(self.value_logits(sequence_inputs), axis=1)
        logits = self.normalize_pi(self.policy_logits(sequence_inputs))
        trimmed_logits, mask_logits = self.apply_pi_mask(logits, features_window)
        mask_result = trimmed_logits if not apply_mask else mask_logits
        if call_print is True:
            print(
                "call took : {0:03f} for batch of {1:03}".format(
                    time.time() - start, batch_size
                )
            )
        return logits, values, mask_result

    def apply_pi_mask(
        self, logits, features_window: MathyInputsType,
    ):
        """Take the policy_mask from a batch of features and multiply
        the policy logits by it to remove any invalid moves"""
        mask = features_window[MathyInputs.mask]
        logits_shape = tf.shape(logits)
        features_mask = tf.reshape(
            mask, (logits_shape[0], -1, self.predictions), name="pi_mask_reshape"
        )
        features_mask = tf.cast(features_mask, dtype=tf.float32)
        # Trim the logits to match the feature mask
        trim_logits = logits[:, : features_mask.shape[1], :]
        trim_logits_shape = tf.shape(trim_logits)
        mask_logits = tf.multiply(trim_logits, features_mask, name="mask_logits")
        negative_mask_logits = tf.where(
            tf.equal(mask_logits, tf.constant(0.0)),
            tf.fill(trim_logits_shape, -1000000.0),
            mask_logits,
            name="softmax_negative_logits",
        )
        return trim_logits, negative_mask_logits

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


def get_or_create_policy_model(
    args: BaseConfig,
    env_actions: int,
    initial_state: Optional[MathyWindowObservation] = None,
) -> PolicyValueModel:

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, args.model_name)
    # Transfer weights/optimizer from a different model
    if args.init_model_from is not None:
        if args.model_format == "keras":
            init_model_path = os.path.join(args.init_model_from, args.model_name)
            opt = f"{init_model_path}.optimizer"
            mod = f"{init_model_path}.h5"
            if not os.path.exists(f"{model_path}.h5"):
                if os.path.exists(opt) and os.path.exists(mod):
                    print(f"initialize model weights from: {init_model_path}")
                    copyfile(mod, model_path)
                    copyfile(opt, model_path)
            else:
                raise ValueError(
                    f"model already exists at: {model_path}, cannot initialize"
                )
            if os.path.exists(opt) and os.path.exists(mod):
                print(f"initialize model from: {init_model_path}")
                copyfile(opt, model_path)
                copyfile(mod, model_path)
            else:
                raise ValueError(
                    f"model already exists at: {model_path}, cannot initialize"
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
    model.compile(
        optimizer=model.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    # else:
    #     model = PolicyValueModel(args=args, predictions=env_actions)

    if initial_state is not None:
        # NOTE: This is needed to properly initialize the trainable vars
        #       for the global model. Each prediction path must be called
        #       here or you'll get gradient descent errors about variables
        #       not matching gradients when the local_model tries to optimize
        #       against the global_model.
        model.predict(initial_state.to_inputs())
        model.build(initial_state.to_input_shapes())

    if args.model_format == "keras":
        opt = f"{model_path}.optimizer"
        mod = f"{model_path}.h5"
        if os.path.exists(mod):
            if args.verbose:
                print(f"Loading keras model and optimizer from: {mod}, {opt}")
            model.load_weights(mod)
            model._make_train_function()
            with open(opt, "rb") as f:
                weight_values = pickle.load(f)

            model.optimizer.set_weights(weight_values)
    elif args.model_format == "tf":
        if os.path.exists(os.path.join(args.model_dir, "checkpoint")):
            if args.verbose:
                print("Loading model from: {}".format(model_path))
            model.load_weights(model_path)
    else:
        raise ValueError(f"error: unknown model format ({args.model_format})")

    return model
