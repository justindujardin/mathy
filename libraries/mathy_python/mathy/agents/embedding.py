from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf

from mathy.agents.base_config import BaseConfig
from mathy.core.expressions import MathTypeKeysMax
from mathy.state import (
    MathyInputsType,
    MathyWindowObservation,
    ObservationFeatureIndices,
)
from mathy.agents.densenet import DenseNetStack

from .attention import SeqSelfAttention


class MathyEmbedding(tf.keras.Model):
    def __init__(self, config: BaseConfig, **kwargs):
        super(MathyEmbedding, self).__init__(**kwargs)
        self.config = config
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=MathTypeKeysMax,
            output_dim=self.config.embedding_units,
            name="nodes_input",
            mask_zero=True,
        )
        # +1 for the value
        # +2 for the problem type hashes
        self.concat_size = 3
        self.values_dense = tf.keras.layers.Dense(
            self.config.units, name="values_input"
        )
        self.type_dense = tf.keras.layers.Dense(self.config.units, name="type_input")
        self.in_dense = tf.keras.layers.Dense(
            self.config.units,
            # In transform gets the embeddings concatenated with the
            # floating point value at each node.
            input_shape=(None, self.config.embedding_units + self.concat_size),
            name="in_dense",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.output_dense = tf.keras.layers.Dense(
            self.config.embedding_units,
            name="out_dense",
            activation="relu",
            kernel_initializer="he_normal",
        )

        NormalizeClass = tf.keras.layers.LayerNormalization
        if self.config.normalization_style == "batch":
            NormalizeClass = tf.keras.layers.BatchNormalization
        self.out_dense_norm = NormalizeClass(name="out_dense_norm")
        self.nodes_lstm_norm = NormalizeClass(name="lstm_nodes_norm")
        self.time_lstm_norm = NormalizeClass(name="time_lstm_norm")
        self.lstm_nodes = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.config.lstm_units,
                name="nodes_lstm",
                time_major=False,
                return_sequences=True,
                dropout=self.config.dropout,
            ),
            merge_mode="ave",
        )
        self.lstm_time = tf.keras.layers.LSTM(
            self.config.lstm_units,
            name="time_lstm",
            time_major=True,
            return_sequences=True,
            dropout=self.config.dropout,
        )
        self.lstm_attention = SeqSelfAttention(
            attention_activation="sigmoid",
            name="self_attention",
            return_attention=True,
        )

    def call(self, features: MathyInputsType, train: tf.Tensor = None) -> tf.Tensor:
        output = tf.concat(
            [
                self.token_embedding(features[ObservationFeatureIndices.nodes]),
                self.values_dense(
                    tf.expand_dims(features[ObservationFeatureIndices.values], axis=-1)
                ),
                self.type_dense(features[ObservationFeatureIndices.type]),
            ],
            axis=-1,
            name="input_vectors",
        )
        output = self.in_dense(output)
        output = self.lstm_time(output)
        output = self.time_lstm_norm(output)
        output = self.lstm_nodes(output)
        output = self.nodes_lstm_norm(output)
        output, attention = self.lstm_attention(output)
        return self.out_dense_norm(self.output_dense(output)), attention
