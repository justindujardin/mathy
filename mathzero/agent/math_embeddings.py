import sys
import collections
import os
import time
import random
import numpy
import math
import sys
from multiprocessing import cpu_count
from itertools import zip_longest
from lib.progress.bar import Bar
from lib.average_meter import AverageMeter

from mathzero.model.math_estimator import math_estimator
from mathzero.environment_state import MathEnvironmentState
from mathzero.model.math_predictor import MathPredictor
from mathzero.agent.math_embeddings_estimator import embeddings_estimator
from mathzero.model.features import (
    pad_array,
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_PROBLEM_TYPE,
    FEATURE_COLUMNS,
)


class NetConfig:
    def __init__(
        self, lr=0.001, dropout=0.2, epochs=4, batch_size=256, log_frequency=250
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_frequency = log_frequency


class MathEmbeddings:
    def __init__(self, num_actions, model_dir, embedding_dimensions):
        import tensorflow as tf

        self.model_dir = model_dir
        self.embedding_dimensions = 32

        session_config = tf.compat.v1.ConfigProto()
        estimator_config = tf.estimator.RunConfig(session_config=session_config)
        self.action_size = num_actions
        self.args = NetConfig()
        self.f_move_count = tf.feature_column.numeric_column(
            key=FEATURE_MOVE_COUNTER, dtype=tf.int16
        )
        self.f_moves_remaining = tf.feature_column.numeric_column(
            key=FEATURE_MOVES_REMAINING, dtype=tf.int16
        )
        self.f_node_count = tf.feature_column.numeric_column(
            key=FEATURE_NODE_COUNT, dtype=tf.int16
        )
        self.f_problem_type = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=FEATURE_PROBLEM_TYPE, num_buckets=32
            )
        )
        self.feature_columns = [
            self.f_problem_type,
            self.f_node_count,
            self.f_move_count,
            self.f_moves_remaining,
        ]

        #
        # Sequence features
        #
        self.f_token_types_sequence = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_hash_bucket(
                key=FEATURE_TOKEN_TYPES, hash_bucket_size=24, dtype=tf.int8
            ),
            dimension=32,
        )
        self.f_token_values_sequence = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_hash_bucket(
                key=FEATURE_TOKEN_VALUES, hash_bucket_size=128, dtype=tf.string
            ),
            dimension=32,
        )
        self.sequence_columns = [
            self.f_token_types_sequence,
            self.f_token_values_sequence,
        ]

        #
        # Estimator
        #
        self.network = tf.estimator.Estimator(
            config=estimator_config,
            model_fn=embeddings_estimator,
            model_dir=model_dir,
            params={
                "feature_columns": self.feature_columns,
                "sequence_columns": self.sequence_columns,
                "action_size": self.action_size,
                "embedding_dimensions": self.embedding_dimensions,
                "batch_size": self.args.batch_size,
            },
        )
        self._worker = MathPredictor(self.network, self.args)

    def predict(self, env_state: MathEnvironmentState):
        input_features = env_state.to_input_features(return_batch=True)
        # start = time.time()
        prediction = self._worker.predict(input_features)
        result = prediction["embedding"]
        # print("predict : {0:03f}".format(time.time() - start))
        # print("focus is : {0:03f}".format(prediction["out_focus"][0]))
        return result

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()

