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
from tf_siren import SinusodialRepresentationDense, SIRENModel
from wasabi import msg

from ..core.expressions import MathTypeKeysMax
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
from .config import AgentConfig


def build_agent_model(
    config: AgentConfig, predictions: int, name="embeddings"
) -> tf.keras.Model:
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
    values_dense = SinusodialRepresentationDense(config.units, name="values_input")
    type_dense = SinusodialRepresentationDense(config.units, name="type_input")
    time_dense = SinusodialRepresentationDense(config.units, name="time_input")
    siren_mlp = SIRENModel(
        units=config.units, final_units=config.units, num_layers=2, name="siren",
    )
    policy_net = tf.keras.Sequential(
        [
            tf.keras.layers.TimeDistributed(
                SinusodialRepresentationDense(
                    predictions, name="policy_ts_hidden", activation=None,
                ),
                name="policy_logits",
            ),
            tf.keras.layers.LayerNormalization(name="policy_layer_norm"),
        ],
        name="policy_head",
    )
    value_net = tf.keras.Sequential(
        [
            SinusodialRepresentationDense(config.units, name="value_hidden"),
            SinusodialRepresentationDense(1, name="value_logits", activation=None),
        ],
        name="value_head",
    )
    reward_net = tf.keras.Sequential(
        [
            SinusodialRepresentationDense(
                config.units, name="reward_hidden", activation="relu",
            ),
            tf.keras.layers.LayerNormalization(name="reward_layer_norm"),
            SinusodialRepresentationDense(1, name="reward_logits", activation=None,),
        ],
        name="reward_head",
    )
    # Model
    sequence_inputs = siren_mlp(
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
    sequence_mean = tf.reduce_mean(sequence_inputs, axis=1)
    values = value_net(sequence_mean)
    reward_logits = reward_net(sequence_mean)
    logits = policy_net(sequence_inputs)
    inputs = [
        nodes_in,
        values_in,
        type_in,
        time_in,
    ]
    outputs = [logits, values, reward_logits]
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


AgentModel = tf.keras.Model


def _load_model(model_path: Path, predictions: int) -> AgentModel:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "SIRENModel": SIRENModel,
            "SinusodialRepresentationDense": SinusodialRepresentationDense,
        },
    )
    model.opt = model.optimizer
    model.predictions = predictions
    return model


def get_or_create_policy_model(
    config: AgentConfig,
    predictions: int,
    is_main=False,
    required=False,
    env: MathyEnv = None,
) -> AgentModel:
    if env is None:
        env = PolySimplify()
    observation: MathyObservation = env.state_to_observation(env.get_initial_state()[0])
    initial_state: MathyWindowObservation = observations_to_window([observation])

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


def load_policy_value_model(
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
    initial_state: MathyWindowObservation = observations_to_window([observation])
    init_inputs = initial_state.to_inputs()
    init_shapes = initial_state.to_input_shapes()
    model: AgentModel = build_agent_model(
        config=args, predictions=env.action_size, name="agent"
    )
    model.compile(optimizer=model.opt, loss="mse", metrics=["accuracy"])
    model.build(init_shapes)
    model.summary()
    model.predict(init_inputs)

    if not silent:
        with msg.loading(f"Loading model: {model_file}..."):
            _load_model(model_file, env.action_size)
        msg.good(f"Loaded model: {model_file}")
    else:
        _load_model(model_file, env.action_size)
    return model, args
