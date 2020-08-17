import os
from pathlib import Path
from typing import Tuple

import srsly
import tensorflow as tf
from tensorflow.keras import backend as K
from wasabi import msg

from mathy_core.expressions import MathTypeKeysMax
from ..env import MathyEnv
from ..envs import PolySimplify
from ..state import (
    MathyInputsType,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from mathy_core.util import print_error
from .config import AgentConfig


AgentModel = tf.keras.Model


@tf.function()
def call_model(model: AgentModel, inputs: MathyInputsType):
    return model.call(inputs)


def build_agent_model(
    config: AgentConfig = None, predictions: int = 6, name="embeddings"
) -> tf.keras.Model:
    if config is None:
        config = AgentConfig()
    nodes_in = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="nodes_in")
    values_in = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name="values_in")
    type_in = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32, name="type_in")
    time_in = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32, name="time_in")
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
            tf.keras.layers.Dense(predictions, name="policy_logits", activation=None,),
            tf.keras.layers.LayerNormalization(name="policy_layer_norm"),
        ],
        name="policy_head",
    )
    value_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(config.units, name="value_hidden"),
            tf.keras.layers.Dense(1, name="value_logits", activation=None),
        ],
        name="value_head",
    )
    reward_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units, name="reward_hidden", activation="relu",
            ),
            tf.keras.layers.LayerNormalization(name="reward_layer_norm"),
            tf.keras.layers.Dense(1, name="reward_logits", activation=None),
        ],
        name="reward_head",
    )
    # Action parameters head
    params_in = tf.keras.layers.Dense(predictions, name="node_param_in")
    params_net = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(config.max_len, name="node_param_hidden"),
        name="node_param_logits",
    )

    # Model
    sequence_inputs = in_dense(
        tf.concat(
            [
                token_embedding(nodes_in),
                values_dense(tf.expand_dims(values_in, axis=-1)),
                type_dense(type_in),
                time_dense(time_in),
            ],
            axis=-1,
            name="input_vectors",
        )
    )
    output = lstm_time(sequence_inputs)
    output = time_lstm_norm(output)
    output = lstm_nodes(output)
    output = nodes_lstm_norm(output)
    params = params_net(tf.expand_dims(params_in(output), axis=-1))
    # squeezed = tf.reduce_max(output, axis=1)
    squeezed = output
    values = value_net(squeezed)
    reward_logits = reward_net(squeezed)
    logits = policy_net(output)
    inputs = [
        nodes_in,
        values_in,
        type_in,
        time_in,
    ]
    outputs = [logits, params, values, reward_logits]
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


def _load_model(model_path: Path, predictions: int) -> AgentModel:
    model = tf.keras.models.load_model(str(model_path))
    model.opt = model.optimizer
    model.predictions = predictions
    return model


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
    if model_path.is_dir():
        if is_main and config.verbose:
            with msg.loading(f"Loading model: {model_path}..."):
                model = _load_model(model_path, predictions)
            msg.good(f"Loaded model: {model_path}")
        else:
            model = _load_model(model_path, predictions)
    elif required:
        print_error(
            ValueError("Model Not Found"),
            f"Cannot find model: {model_path}",
            print_error=False,
        )
    elif is_main:
        cfg = f"{config.model_dir}{os.path.sep}model.config.json"
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
    model_file = Path(model_data_folder) / args.model_name
    if not model_file.exists():
        raise ValueError(f"model not found: {model_file}")
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
        with msg.loading(f"Loading model: {model_file}..."):
            _load_model(model_file, env.action_size)
        msg.good(f"Loaded model: {model_file}")
        model.summary()
    else:
        _load_model(model_file, env.action_size)
    return model, args
