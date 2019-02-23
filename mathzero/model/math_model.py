import sys
import collections
import os
import time
import random
import numpy
import math
import sys
from itertools import zip_longest
from lib.progress.bar import Bar
from lib.average_meter import AverageMeter

from mathzero.model.math_estimator import math_estimator
from mathzero.environment_state import MathEnvironmentState
from mathzero.model.math_predictor import MathPredictor
from mathzero.model.features import (
    parse_examples_for_training,
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_MOVE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_COLUMNS,
)


class NetConfig:
    def __init__(
        self, lr=0.0001, dropout=0.2, max_steps=10000, batch_size=256, log_frequency=250
    ):
        self.lr = lr
        self.dropout = dropout
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_frequency = log_frequency


class MathModel:
    def __init__(self, game, model_dir, all_memory=False, dev_mode=False):
        import tensorflow as tf

        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = (
            game.get_gpu_fraction()
        )
        session_config.gpu_options.allow_growth = True
        estimator_config = tf.estimator.RunConfig(session_config=session_config)
        self.action_size = game.get_agent_actions_count()

        self.args = NetConfig()
        # Feature columns describe how to use the input.
        self.token_value_feature = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key=FEATURE_TOKEN_VALUES, hash_bucket_size=12
            ),
            dimension=32,
        )
        self.feature_tokens_type = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key=FEATURE_TOKEN_TYPES, hash_bucket_size=12, dtype=tf.int64
            ),
            dimension=4,
        )

        self.feature_move_count = tf.feature_column.numeric_column(
            key=FEATURE_MOVE_COUNT, dtype=tf.int16
        )
        self.feature_node_count = tf.feature_column.numeric_column(
            key=FEATURE_NODE_COUNT, dtype=tf.int16
        )
        self.feature_problem_type = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key=FEATURE_PROBLEM_TYPE, num_buckets=32
            )
        )
        self.feature_columns = [
            self.feature_problem_type,
            self.feature_tokens_type,
            self.feature_node_count,
            self.feature_move_count,
            self.token_value_feature,
        ]
        self.network = tf.estimator.Estimator(
            config=estimator_config,
            model_fn=math_estimator,
            model_dir=model_dir,
            params={
                "feature_columns": self.feature_columns,
                "action_size": self.action_size,
                "learning_rate": self.args.lr,
                "hidden_units": [4, 4],
            },
        )
        self._worker = MathPredictor(self.network, self.args)

    def train(self, examples):
        """examples: list of examples in JSON format"""
        from .math_hooks import TrainingLoggerHook, TrainingEarlyStopHook
        import tensorflow as tf

        print(
            "Training model for up to ({}) steps with ({}) examples...".format(
                self.args.max_steps, len(examples)
            )
        )
        self.network.train(
            hooks=[
                TrainingEarlyStopHook(),
                TrainingLoggerHook(self.args.batch_size, self.args.log_frequency),
            ],
            steps=self.args.max_steps,
            input_fn=lambda: parse_examples_for_training(examples),
        )
        return True

    def predict(self, env_state: MathEnvironmentState):
        tokens = env_state.parser.tokenize(env_state.agent.problem)
        types = []
        values = []
        for t in tokens:
            types.append(t.type)
            values.append(t.value)
        input_features = {
            FEATURE_TOKEN_TYPES: [types],
            FEATURE_TOKEN_VALUES: [values],
            FEATURE_NODE_COUNT: [len(values)],
            FEATURE_MOVE_COUNT: [env_state.agent.moves_remaining],
            FEATURE_PROBLEM_TYPE: [env_state.agent.problem_type],
        }
        start = time.time()
        prediction = self._worker.predict(input_features)
        # print("predict : {0:03f}".format(time.time() - start))
        # print("focus is : {0:03f}".format(prediction["out_focus"][0]))
        return (
            prediction["out_policy"],
            prediction["out_value"][0],
            prediction["out_focus"][0],
        )

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()