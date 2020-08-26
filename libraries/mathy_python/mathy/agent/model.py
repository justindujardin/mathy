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
from .config import AgentConfig
from .episode_memory import EpisodeMemory
from .trfl.discrete_policy_gradient_ops import sequence_advantage_actor_critic_loss


class AgentModel(tf.keras.Model):
    predictions: int
    opt: tf.keras.optimizers.Optimizer


MathyOutputsType = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


@tf.function()
def call_model(model: AgentModel, inputs: MathyInputsType):
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
    nodes_lstm_norm = tf.keras.layers.LayerNormalization(name="lstm_nodes_norm")
    time_lstm_norm = tf.keras.layers.LayerNormalization(name="time_lstm_norm")
    lstm_nodes = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.lstm_units,
            name="nodes_lstm",
            time_major=False,
            return_sequences=False,
            dropout=config.dropout,
        ),
        merge_mode="ave",
    )
    lstm_time = tf.keras.layers.LSTM(
        config.lstm_units,
        name="time_lstm",
        time_major=True,
        return_sequences=True,
        dropout=config.dropout,
    )
    policy_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units, name="fn_head/hidden", activation="swish"
            ),
            tf.keras.layers.Dense(predictions, name="fn_head/logits", activation=None),
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
            tf.keras.layers.Dense(
                predictions, name="params_head/actions", activation="swish"
            ),
            tf.keras.layers.Reshape((predictions, 1)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(config.max_len, name="params_head/params"),
            ),
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
    out_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.lr_initial,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=config.lr_decay_staircase,
    )
    out_model.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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


def seq_len_from_observation(observation: MathyObservation) -> int:
    return len([n for n in observation.nodes if n != 0])


def compute_agent_loss(
    model: AgentModel,
    args: AgentConfig,
    episode_memory: EpisodeMemory,
    bootstrap_value: float = 0.0,
    gamma: float = 0.99,
) -> Tuple[AgentLosses, MathyOutputsType]:
    """Compute the policy/value/reward/entropy losses for an episode given its
    current memory of the episode steps."""
    batch_size = len(episode_memory.actions)
    inputs = episode_memory.to_episode_window().to_inputs()
    model_results = call_model(model, inputs)
    logits, params, values = model_results
    rewards_tensor = tf.convert_to_tensor(
        [[r] for r in episode_memory.rewards], dtype=tf.float32
    )
    pcontinues = tf.convert_to_tensor([[gamma]] * batch_size, dtype=tf.float32)
    bootstrap_value_tensor = tf.convert_to_tensor([bootstrap_value], dtype=tf.float32)
    policy_fn_labels = tf.convert_to_tensor([[a[0]] for a in episode_memory.actions])
    a3c_fn_loss = sequence_advantage_actor_critic_loss(
        policy_logits=tf.expand_dims(logits, axis=1),
        baseline_values=values,
        actions=policy_fn_labels,
        rewards=rewards_tensor,
        pcontinues=pcontinues,
        bootstrap_value=bootstrap_value_tensor,
        normalise_entropy=True,
        entropy_cost=args.policy_fn_entropy_cost,
    )

    # Select the tensor rows that correspond to the chosen actions
    # at each timestep. This way we only calculate loss on the args
    # logits associated with the action that was taken.
    args_logits_stack = []
    args_labels = []
    for i in range(batch_size):
        param_logits = params[i]
        rule, node = episode_memory.actions[i]
        args_logits_stack.append(param_logits[rule])
        args_labels.append([node])
    args_logits = tf.stack(args_logits_stack)
    args_labels = tf.convert_to_tensor(args_labels)
    a3c_args_loss = sequence_advantage_actor_critic_loss(
        policy_logits=tf.expand_dims(args_logits, axis=1),
        baseline_values=values,
        actions=args_labels,
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
    total_loss = (
        tf.squeeze(a3c_fn_loss.loss)
        # + value_loss
        # + fn_entropy_loss
        # + args_policy_loss
        # + args_entropy_loss
    )
    losses = AgentLosses(
        fn_policy=fn_policy_loss,
        fn_entropy=fn_entropy_loss,
        args_policy=args_policy_loss,
        args_entropy=args_entropy_loss,
        value=value_loss,
        total=total_loss,
    )
    return losses, model_results


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
    memory = EpisodeMemory(config.max_len)
    memory.store(observation=observation, action=(0, 0), reward=0.1, value=1.0)
    with tf.GradientTape() as tape:
        losses: AgentLosses
        losses, _ = compute_agent_loss(model, config, memory)
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
