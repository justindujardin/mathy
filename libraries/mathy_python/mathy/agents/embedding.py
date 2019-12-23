import json
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
from pydantic import BaseModel

from mathy.agents.base_config import BaseConfig
from mathy.core.expressions import MathTypeKeysMax
from mathy.state import (
    MathyInputsType,
    MathyWindowObservation,
    ObservationFeatureIndices,
)


class EmbeddingsState(object):
    """Persistent Embedding RNN state for functional keras model"""

    config: BaseConfig
    state_h: tf.Variable
    state_c: tf.Variable
    state_history_h: tf.Variable

    def __init__(
        self,
        config: BaseConfig,
        state_h: tf.Variable,
        state_c: tf.Variable,
        state_history_h: tf.Variable,
    ):
        self.config = config
        self.state_h = state_h
        self.state_c = state_c
        self.state_history_h = state_history_h

    def reset_state(self, force: bool = False):
        """Zero out the RNN state for a new episode"""
        if self.config.episode_reset_state_h or force is True:
            self.state_h.assign(tf.zeros([1, self.config.lstm_units]))
            self.state_history_h.assign(tf.zeros([1, self.config.lstm_units * 2]))
        if self.config.episode_reset_state_c or force is True:
            self.state_c.assign(tf.zeros([1, self.config.lstm_units]))


def mathy_embedding(
    units: int = 64,
    lstm_units: int = 128,
    embedding_units: int = 128,
    use_env_features: bool = False,
    use_node_values: bool = True,
    name="mathy_embedding",
    return_state=False,
) -> Union[Tuple[tf.keras.Model, EmbeddingsState], tf.keras.Model]:
    from mathy.envs import PolySimplify
    from mathy.agents.base_config import BaseConfig

    args = BaseConfig(
        use_env_features=use_env_features, use_node_values=use_node_values
    )
    env = PolySimplify()
    state, _ = env.get_initial_state()
    window: MathyWindowObservation = state.to_empty_window(1, args.lstm_units)
    model, state = build_model(args)
    model.build(window.to_input_shapes())
    inputs = window.to_inputs_tiled()
    model.predict(inputs)
    return model if not return_state else (model, state)


def mathy_embedding_with_env_features():
    return mathy_embedding(use_env_features=True)


def build_model(config: BaseConfig) -> Tuple[tf.keras.Model, EmbeddingsState]:
    nodes_in = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="nodes_in")
    values_in = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name="values_in")
    type_in = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32, name="type_in")
    time_in = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32, name="time_in")
    rnn_state_h_in = tf.keras.layers.Input(
        shape=(config.lstm_units,), dtype=tf.float32, name="rnn_state_h_in",
    )
    rnn_state_c_in = tf.keras.layers.Input(
        shape=(config.lstm_units,), dtype=tf.float32, name="rnn_state_c_in",
    )
    rnn_history_h_in = tf.keras.layers.Input(
        batch_shape=(1, config.lstm_units), dtype=tf.float32, name="rnn_history_h_in",
    )

    state = EmbeddingsState(
        config=config,
        state_c=tf.Variable(
            tf.zeros([1, config.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_c",
        ),
        state_h=tf.Variable(
            tf.zeros([1, config.lstm_units]),
            trainable=False,
            name="embedding/rnn/agent_state_h",
        ),
        state_history_h=tf.Variable(
            tf.zeros([1, config.lstm_units * 2]),
            trainable=False,
            name="embedding/rnn/agent_state_h_with_history",
        ),
    )

    token_embedding = tf.keras.layers.Embedding(
        input_dim=MathTypeKeysMax,
        output_dim=config.embedding_units,
        name="nodes_embedding",
        mask_zero=True,
    )

    # +1 for the value
    # +1 for the time
    # +2 for the problem type hashes
    concat_size = 4 if config.use_env_features else 1 if config.use_node_values else 0
    # if use_node_values:
    #     values_dense = tf.keras.layers.Dense(units, name="values_input")
    if config.use_env_features:
        time_dense = tf.keras.layers.Dense(config.units, name="time_input")
        type_dense = tf.keras.layers.Dense(config.units, name="type_input")
    in_dense = tf.keras.layers.Dense(
        config.units,
        # In transform gets the embeddings concatenated with the
        # floating point value at each node.
        input_shape=(None, config.embedding_units + concat_size),
        name="in_dense",
        activation="relu",
        kernel_initializer="he_normal",
    )
    out_dense_norm = tf.keras.layers.LayerNormalization(name="out_dense_norm")
    time_lstm_norm = tf.keras.layers.LayerNormalization(name="lstm_time_norm")
    nodes_lstm_norm = tf.keras.layers.LayerNormalization(name="lstm_nodes_norm")
    time_lstm = tf.keras.layers.LSTM(
        config.lstm_units,
        name="timestep_lstm",
        return_sequences=True,
        time_major=True,
        return_state=True,
    )
    nodes_lstm = tf.keras.layers.LSTM(
        config.lstm_units,
        name="nodes_lstm",
        return_sequences=True,
        time_major=False,
        return_state=False,
    )
    output_dense = tf.keras.layers.Dense(
        config.embedding_units,
        name="out_dense",
        activation="relu",
        kernel_initializer="he_normal",
    )
    # Model
    values_shape = tf.shape(values_in, name="get_dynamic_shape")
    batch_size = tf.squeeze(values_shape[0])
    sequence_length = tf.squeeze(values_shape[1])

    values = tf.expand_dims(values_in, axis=-1, name="values_input")
    query = token_embedding(nodes_in)

    # if config.use_node_values:
    #     values = values_dense(values)

    # If not using env features, only concatenate the tokens and values
    if not config.use_env_features and config.use_node_values:
        query = tf.concat([query, values], axis=-1, name="tokens_and_values")
        query.set_shape((None, None, config.embedding_units + concat_size))
    elif config.use_env_features:
        env_inputs = [
            query,
            type_dense(type_in),
            time_dense(time_in),
        ]
        if config.use_node_values:
            env_inputs.insert(1, values)

        query = tf.concat(env_inputs, axis=-1, name="tokens_and_values",)
        # query.set_shape((None, None, self.embedding_units + self.concat_size))

    # Input dense transforms
    query = in_dense(query)
    query = nodes_lstm(query)
    query = nodes_lstm_norm(query)
    query, state_h, state_c = time_lstm(
        query, initial_state=[rnn_state_h_in, rnn_state_c_in]
    )
    query = time_lstm_norm(query)
    state.state_h.assign(state_h[-1:])
    state.state_c.assign(state_c[-1:])

    # Concatenate the RNN output state with our historical RNN state
    # See: https://arxiv.org/pdf/1810.04437.pdf
    rnn_state_with_history = tf.concat(
        [state_h[-1:], rnn_history_h_in], axis=-1, name="state_h_with_history"
    )
    state.state_history_h.assign(rnn_state_with_history)
    lstm_context = tf.tile(
        tf.expand_dims(rnn_state_with_history, axis=0),
        [batch_size, sequence_length, 1],
        name="lstm_context",
    )
    # Combine the output LSTM states with the historical average LSTM states
    # and concatenate it with the query
    output = tf.concat([query, lstm_context], axis=-1, name="combined_outputs")

    final_out = out_dense_norm(output_dense(output))

    # Result
    inputs = [
        nodes_in,
        values_in,
        rnn_state_h_in,
        rnn_state_c_in,
        rnn_history_h_in,
    ]
    if config.use_env_features:
        inputs += [type_in, time_in]
    out_model = tf.keras.Model(inputs=inputs, outputs=[final_out],)

    return (
        out_model,
        state,
    )
