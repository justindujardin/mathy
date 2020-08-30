import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import srsly
import tensorflow as tf
from mathy_core.expressions import MathTypeKeysMax
from mathy_core.util import print_error
from tensorflow.keras import backend as K
from wasabi import msg

from ..env import MathyEnv
from ..envs import PolySimplify
from ..state import (
    MathyInputsType,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from ..types import ActionType
from .config import AgentConfig
from .episode_memory import EpisodeMemory
from .trfl.discrete_policy_gradient_ops import sequence_advantage_actor_critic_loss


class AgentModel(tf.keras.Model):
    predictions: int
    opt: tf.keras.optimizers.Optimizer


MathyOutputsType = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


@tf.function()
def call_model(model: AgentModel, inputs: MathyInputsType):
    # print(len(inputs["nodes_in"]))
    return model.call(inputs)


def build_agent_model(
    config: AgentConfig = None, predictions: int = 6, name="embeddings"
) -> AgentModel:
    if config is None:
        config = AgentConfig()
    nodes_in = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="nodes_in")
    values_in = tf.keras.layers.Input(
        shape=(config.max_len, 1), dtype=tf.float32, name="values_in"
    )
    type_in = tf.keras.layers.Input(
        shape=(config.max_len, 2), dtype=tf.float32, name="type_in"
    )
    time_in = tf.keras.layers.Input(
        shape=(config.max_len, 1), dtype=tf.float32, name="time_in"
    )
    token_embedding = tf.keras.layers.Embedding(
        input_dim=MathTypeKeysMax,
        output_dim=config.embedding_units,
        name="nodes_input",
        mask_zero=True,
    )
    values_dense = tf.keras.layers.Dense(
        config.units, name="values_input", activation="swish",
    )
    type_dense = tf.keras.layers.Dense(
        config.units, name="type_input", activation="swish",
    )
    time_dense = tf.keras.layers.Dense(
        config.units, name="time_input", activation="swish"
    )
    in_dense = tf.keras.layers.Dense(
        config.units,
        name="in_dense",
        activation="swish",
        kernel_initializer="he_normal",
    )
    in_norm = tf.keras.layers.LayerNormalization(name="input_norm")
    nodes_lstm_norm = tf.keras.layers.LayerNormalization(name="lstm_nodes_norm")
    time_lstm_norm = tf.keras.layers.LayerNormalization(name="time_lstm_norm")
    lstm_nodes = tf.keras.layers.LSTM(
        config.lstm_units, name="nodes_lstm", time_major=False, return_sequences=False,
    )
    lstm_time = tf.keras.layers.LSTM(
        config.lstm_units, name="time_lstm", time_major=True, return_sequences=True,
    )
    # TODO: set output layers initializer to zero (or near) to maximize entropy
    # see: https://youtu.be/8EcdaCk9KaQ?t=1812
    policy_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units, name="fn_head/hidden", activation="swish"
            ),
            tf.keras.layers.Dense(predictions, name="fn_head/logits", activation=None,),
            tf.keras.layers.LayerNormalization(name="fn_head/logits_layer_norm"),
        ],
        name="fn_head",
    )
    value_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(config.units, name="value_h1", activation="swish"),
            tf.keras.layers.Dense(config.units, name="value_h2", activation="swish"),
            tf.keras.layers.Dense(1, name="value_logits", activation=None),
        ],
        name="value_head",
    )
    # Action parameters head
    params_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units, name="params_head/hidden", activation="swish"
            ),
            tf.keras.layers.Dense(config.max_len, name="params_head/params"),
            tf.keras.layers.LayerNormalization(name="params_head/logits_layer_norm"),
        ],
        name="params_head",
    )

    # Model
    sequence_inputs = in_dense(
        tf.concat(
            [
                token_embedding(nodes_in),
                values_dense(values_in),
                type_dense(type_in),
                time_dense(time_in),
            ],
            axis=-1,
            name="input_vectors",
        )
    )
    sequence_inputs = in_norm(sequence_inputs)
    lstm_output = lstm_time(sequence_inputs)
    lstm_output = time_lstm_norm(lstm_output)
    lstm_output = lstm_nodes(lstm_output)
    lstm_output = nodes_lstm_norm(lstm_output)
    fn_policy_logits = policy_net(lstm_output)
    params = params_net(
        tf.concat([lstm_output, fn_policy_logits], axis=-1, name="params_in")
    )
    values = value_net(lstm_output)
    inputs = [
        nodes_in,
        values_in,
        type_in,
        time_in,
    ]
    outputs = [fn_policy_logits, params, values]
    out_model = AgentModel(inputs=inputs, outputs=outputs, name=name)
    out_model.opt = tf.keras.optimizers.Adam(learning_rate=config.lr)
    out_model.predictions = predictions
    return out_model


@dataclass
class AgentLosses:
    """Agent losses individually and in total"""

    fn_policy: tf.Tensor
    fn_entropy: tf.Tensor
    args_entropy: tf.Tensor
    args_policy: tf.Tensor
    value: tf.Tensor
    total: tf.Tensor

    @classmethod
    def zero(cls) -> "AgentLosses":
        return AgentLosses(
            tf.convert_to_tensor(0.0),
            tf.convert_to_tensor(0.0),
            tf.convert_to_tensor(0.0),
            tf.convert_to_tensor(0.0),
            tf.convert_to_tensor(0.0),
            tf.convert_to_tensor(0.0),
        )

    def accumulate(self, other: "AgentLosses") -> None:
        self.fn_policy = tf.add(self.fn_policy, other.fn_policy)  # type:ignore
        self.fn_entropy = tf.add(self.fn_entropy, other.fn_entropy)  # type:ignore
        self.args_entropy = tf.add(self.args_entropy, other.args_entropy)  # type:ignore
        self.args_policy = tf.add(self.args_policy, other.args_policy)  # type:ignore
        self.value = tf.add(self.value, other.value)  # type:ignore
        self.total = tf.add(self.total, other.total)  # type:ignore


def seq_len_from_observation(observation: MathyObservation) -> int:
    return len([n for n in observation.nodes if n != 0])


def compute_agent_loss(
    model: AgentModel,
    args: AgentConfig,
    inputs: MathyWindowObservation,
    actions: List[ActionType],
    rewards: List[float],
    bootstrap_value: float = 0.0,
    gamma: float = 0.99,
) -> AgentLosses:
    """Compute the policy/value/reward/entropy losses for an episode given its
    current memory of the episode steps."""
    batch_size = len(actions)
    assert (
        len(inputs.nodes) == len(actions) == len(rewards)
    ), f"obs/act/rwd must be the same length, not: ({len(inputs.nodes)}, {len(actions)}, {len(rewards)}"
    logits, params, values = call_model(model, inputs.to_inputs())
    unpadded_length: int = inputs.real_length
    rewards_tensor = tf.convert_to_tensor([[r] for r in rewards], dtype=tf.float32)
    pcontinues = tf.convert_to_tensor([[gamma]] * batch_size, dtype=tf.float32)

    # Unpad the various tensors for loss/labels
    logits = logits[-unpadded_length:, :]
    params = params[-unpadded_length:, :]
    values = values[-unpadded_length:]
    actions = actions[-unpadded_length:]
    rewards_tensor = rewards_tensor[-unpadded_length:]
    pcontinues = pcontinues[-unpadded_length:]

    bootstrap_value_tensor = tf.convert_to_tensor([bootstrap_value], dtype=tf.float32)
    policy_fn_labels = tf.convert_to_tensor([[a[0]] for a in actions])
    policy_fn_logits = tf.expand_dims(logits, axis=1)
    a3c_fn_loss = sequence_advantage_actor_critic_loss(
        policy_logits=policy_fn_logits,
        baseline_values=values,
        actions=policy_fn_labels,
        rewards=rewards_tensor,
        pcontinues=pcontinues,
        bootstrap_value=bootstrap_value_tensor,
        normalise_entropy=True,
        entropy_cost=args.policy_fn_entropy_cost,
    )
    args_fn_labels = tf.convert_to_tensor([[a[1]] for a in actions])
    args_fn_logits = tf.expand_dims(params, axis=1)
    a3c_args_loss = sequence_advantage_actor_critic_loss(
        policy_logits=args_fn_logits,
        baseline_values=values,
        actions=args_fn_labels,
        rewards=rewards_tensor,
        pcontinues=pcontinues,
        bootstrap_value=bootstrap_value_tensor,
        normalise_entropy=True,
        entropy_cost=args.policy_args_entropy_cost,
    )

    fn_policy_loss = tf.squeeze(a3c_fn_loss.extra.policy_gradient_loss)
    fn_entropy_loss = tf.squeeze(a3c_fn_loss.extra.entropy_loss)
    args_policy_loss = tf.squeeze(a3c_args_loss.extra.policy_gradient_loss)
    args_entropy_loss = tf.squeeze(a3c_args_loss.extra.entropy_loss)
    value_loss = tf.squeeze(a3c_fn_loss.extra.baseline_loss)
    total_loss = tf.squeeze(a3c_fn_loss.loss) + args_policy_loss + args_entropy_loss
    losses = AgentLosses(
        fn_policy=fn_policy_loss,
        fn_entropy=fn_entropy_loss,
        args_policy=args_policy_loss,
        args_entropy=args_entropy_loss,
        value=value_loss,
        total=total_loss,
    )
    return losses


def predict_action_value(
    model: AgentModel, inputs: MathyInputsType
) -> Tuple[ActionType, float]:
    """Predict the fn/args policies and value/reward estimates for current timestep."""
    mask = inputs.pop("mask_in")[-1]
    action_logits, args_logits, values = call_model(model, inputs)
    action_logits = action_logits[-1:]
    args_logits = args_logits[-1:]
    values = values[-1:]
    fn_mask = mask.numpy().sum(axis=1).astype("bool").astype("int")
    fn_mask_logits = tf.multiply(action_logits, fn_mask, name="mask_logits")
    # Mask the selected rule functions to remove invalid selections
    fn_masked = tf.where(
        tf.equal(fn_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(action_logits), -1000000.0),
        fn_mask_logits,
        name="fn_softmax_negative_logits",
    )
    fn_probs = tf.nn.softmax(fn_masked).numpy()
    fn_action = int(fn_probs.argmax())
    # Mask the node selection policy for each rule function
    args_mask = tf.cast(mask[fn_action], dtype="float32")
    args_mask_logits = tf.multiply(args_logits, args_mask, name="mask_logits")
    args_masked = tf.where(
        tf.equal(args_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(args_logits), -1000000.0),
        args_mask_logits,
        name="args_negative_softmax_logts",
    )
    args_probs = tf.nn.softmax(args_masked).numpy()
    arg_action = int(args_probs.argmax())
    return (fn_action, arg_action), float(tf.squeeze(values[-1]).numpy())


def _load_model(
    model: AgentModel,
    config: AgentConfig,
    model_file: str,
    optimizer_file: str,
    observation: MathyObservation,
    initial_state: MathyWindowObservation,
) -> AgentModel:
    # Load the optimizer weights. We have to take a step
    # with the optimizer before we can load the weights, or
    # TensorFlow will complain that the # of weights don't match
    # what the optimizer expects. {x_X}
    memory = EpisodeMemory(config.max_len, config.prediction_window_size)
    memory.store(observation=observation, action=(0, 0), reward=0.1, value=1.0)
    with tf.GradientTape() as tape:
        window, actions, rewards = memory.to_non_terminal_training_window(1)
        losses: AgentLosses = compute_agent_loss(
            model=model, args=config, inputs=window, actions=actions, rewards=rewards,
        )
    grads = tape.gradient(losses.total, model.trainable_weights)
    zipped_gradients = zip(grads, model.trainable_weights)
    model.opt.apply_gradients(zipped_gradients)
    with open(optimizer_file, "rb") as f:
        weight_values = pickle.load(f)
    model.opt.set_weights(weight_values)

    # Load the model weights
    model.load_weights(model_file)
    if hasattr(model, "make_train_function"):
        model.make_train_function()
    elif hasattr(model, "_make_train_function"):
        getattr(model, "_make_train_function")()
    model.build(initial_state.to_input_shapes())

    return model


def save_model(model: AgentModel, model_path: Union[str, Path]) -> None:
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True)
    assert hasattr(model, "opt") is True, "model.opt must be set to the optimizer"
    with open(f"{model_path}.optimizer", "wb") as f:
        pickle.dump(model.opt.get_weights(), f)
    model_file = f"{model_path}.h5"
    model.save_weights(model_file, save_format="keras")
    step = model.opt.iterations.numpy()
    print(f"[save] step({step}) model({model_file})")


def get_or_create_agent_model(
    config: AgentConfig,
    predictions: int,
    is_main=False,
    required=False,
    env: MathyEnv = None,
) -> AgentModel:
    if env is None:
        env = PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    initial_state: MathyWindowObservation = observations_to_window(
        [observation], config.max_len
    )
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    model_path: Path = Path(config.model_dir) / config.model_name
    model = build_agent_model(config=config, predictions=predictions, name="agent")
    model.compile(optimizer=model.opt, loss="binary_crossentropy", metrics=["accuracy"])
    model.build(initial_state.to_input_shapes())
    opt = f"{model_path}.optimizer"
    mod = f"{model_path}.h5"
    if os.path.exists(mod):
        if is_main and config.verbose:
            with msg.loading(f"Loading model: {mod}..."):
                _load_model(model, config, mod, opt, observation, initial_state)
            msg.good(f"Loaded model: {mod}")
        else:
            _load_model(model, config, mod, opt, observation, initial_state)
    elif required:
        print_error(
            ValueError("Model Not Found"),
            f"Cannot find model: {mod}",
            print_error=False,
        )
    elif is_main:
        cfg = f"{model_path}.config.json"
        if config.verbose:
            msg.info(f"wrote model config: {cfg}")
        srsly.write_json(cfg, config.dict(exclude_defaults=False))

    return model


def load_agent_model(
    model_data_folder: str, silent: bool = False
) -> Tuple[AgentModel, AgentConfig]:
    meta_file = Path(model_data_folder) / "model.config.json"
    if not meta_file.exists():
        raise ValueError(f"model meta not found: {meta_file}")
    args = AgentConfig(**srsly.read_json(str(meta_file)))
    model_path = Path(model_data_folder) / args.model_name
    opt = f"{model_path}.optimizer"
    mod = f"{model_path}.h5"
    if not os.path.exists(mod):
        raise ValueError(f"model not found: {mod}")
    env: MathyEnv = PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    initial_state: MathyWindowObservation = observations_to_window(
        [observation], args.max_len
    )
    init_inputs = initial_state.to_inputs()
    init_shapes = initial_state.to_input_shapes()
    model: AgentModel = build_agent_model(
        config=args, predictions=env.action_size, name="agent"
    )
    model.compile(optimizer=model.opt, loss="mse", metrics=["accuracy"])
    model.build(init_shapes)
    model.predict(init_inputs)

    if not silent:
        with msg.loading(f"Loading model: {mod}..."):
            _load_model(model, args, mod, opt, observation, initial_state)
        msg.good(f"Loaded model: {mod}")
    else:
        _load_model(model, args, mod, opt, observation, initial_state)
    return model, args
