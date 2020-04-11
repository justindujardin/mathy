import os

import srsly
import tensorflow as tf

from ...core.expressions import MathTypeKeysMax
from ...env import MathyEnv
from ..attention import SeqSelfAttention
from .config import SelfPlayConfig

from tensorflow.keras.utils import CustomObjectScope


def zero_model(config: SelfPlayConfig, predictions: int) -> tf.keras.Sequential:
    nodes_input = tf.keras.Input(shape=(None,), name="nodes")
    embed = tf.keras.layers.Embedding(
        input_dim=MathTypeKeysMax,
        output_dim=config.embedding_units,
        name="nodes_input",
        mask_zero=True,
    )
    x = embed(nodes_input)
    self_attn = SeqSelfAttention(attention_activation="sigmoid", name="self_attn",)
    x = self_attn(x)

    shared_dense = tf.keras.layers.Dense(
        predictions,
        name="policy_ts_hidden",
        kernel_initializer="he_normal",
        activation=None,
    )

    policy_net = tf.keras.Sequential(
        [
            tf.keras.layers.TimeDistributed(shared_dense, name="policy_logits"),
            tf.keras.layers.LayerNormalization(name="policy_layer_norm"),
        ],
        name="policy_head",
    )(x)
    x_mean = tf.reduce_mean(x, axis=1)
    value_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units,
                name="value_hidden",
                kernel_initializer="he_normal",
                activation="relu",
            ),
            tf.keras.layers.LayerNormalization(name="value_layer_norm"),
            tf.keras.layers.Dense(
                1, name="value_logits", kernel_initializer="he_normal", activation=None,
            ),
        ],
        name="value_head",
    )(x_mean)
    reward_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config.units,
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
    )(x_mean)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.lr_initial,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=config.lr_decay_staircase,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = tf.keras.Model(
        inputs=[nodes_input], outputs=[policy_net, value_net, reward_net]
    )
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def zero_save(config: SelfPlayConfig, model: tf.keras.Model) -> None:
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    model_path = os.path.join(config.model_dir, config.model_name)
    model_path += ".h5"
    model.save(model_path)
    step = model.optimizer.iterations.numpy()
    print(f"[save] step({step}) model({model_path})")


def get_zero(
    config: SelfPlayConfig,
    predictions: int,
    is_main=False,
    required=False,
    env: MathyEnv = None,
) -> tf.keras.Sequential:
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    model_path = os.path.join(config.model_dir, config.model_name)
    model_file = f"{model_path}.h5"
    if os.path.exists(model_file):
        with CustomObjectScope({"SeqSelfAttention": SeqSelfAttention}):
            return tf.keras.models.load_model(model_file)
    if is_main:
        cfg = f"{model_path}.config.json"
        srsly.write_json(cfg, config.dict(exclude_defaults=False))
    return zero_model(config, predictions)
