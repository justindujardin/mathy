import sys
import collections
import os
import time
import random
import numpy
import math
import sys
from pathlib import Path
from multiprocessing import cpu_count
from itertools import zip_longest
from lib.progress.bar import Bar
from lib.average_meter import AverageMeter

from .math_estimator import math_estimator
from ..environment_state import MathEnvironmentState
from ..model.math_predictor import MathPredictor
from ..model.features import (
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
        self, lr=0.001, dropout=0.2, epochs=1, batch_size=256, log_frequency=250
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_frequency = log_frequency


class MathModel:
    def __init__(
        self,
        action_size,
        model_dir,
        all_memory=False,
        dev_mode=False,
        init_model_dir=None,
    ):
        import tensorflow as tf

        self.model_dir = model_dir
        self.init_model_dir = init_model_dir
        if Path(self.model_dir).is_dir():
            print(
                "Skipping initialization from model because destination folder exists"
            )
            self.init_model_dir = None
        if self.init_model_dir is not None:
            print(
                "initializing model with trainable variables from: {}".format(
                    self.init_model_dir
                )
            )
        self.embedding_dimensions = 64

        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        estimator_config = tf.estimator.RunConfig(session_config=session_config)
        self.action_size = action_size
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
            warm_start_from=self.init_model_dir,
            config=estimator_config,
            model_fn=math_estimator,
            model_dir=model_dir,
            params={
                "feature_columns": self.feature_columns,
                "sequence_columns": self.sequence_columns,
                "embedding_dimensions": self.embedding_dimensions,
                "action_size": self.action_size,
                "learning_rate": self.args.lr,
                "batch_size": self.args.batch_size,
            },
        )
        self._worker = MathPredictor(self.network, self.args)

    def train(self, short_term_examples, long_term_examples):
        """examples: list of examples in JSON format"""
        from .math_hooks import EpochTrainerHook
        import tensorflow as tf
        from .math_dataset import make_training_input_fn

        # Reflection capacity (how many observations should we train on in this meditation?)
        max_examples = 2048

        # Always sample all of the current episodes observations first
        stm_sample = short_term_examples[:max_examples]
        examples = stm_sample
        ltm_sample = []
        # If there's room left, shuffle the long-term examples and sample
        # the remainder from there.
        if len(examples) < max_examples and len(long_term_examples) > 0:
            remaining_capacity = max_examples - len(examples)

            random.shuffle(long_term_examples)
            ltm_sample = long_term_examples[:remaining_capacity]
            examples = examples + ltm_sample

        print(
            "Mediating on {} observations from recent experience and {} past observations".format(
                len(stm_sample), len(ltm_sample)
            )
        )
        print(
            "Training {} epochs with {} examples and learning rate {}...".format(
                self.args.epochs, len(examples), self.args.lr
            )
        )
        max_steps = len(examples) * self.args.epochs
        self.network.train(
            hooks=[
                EpochTrainerHook(self.args.epochs, len(examples), self.args.batch_size)
            ],
            steps=max_steps,
            input_fn=make_training_input_fn(examples, self.args.batch_size),
        )
        return True

    def predict(self, env_state: MathEnvironmentState):
        input_features = env_state.to_input_features(return_batch=True)
        # start = time.time()
        prediction = self._worker.predict(input_features)
        # print("predict : {0:03f}".format(time.time() - start))
        # print("distribution is : {}".format(prediction[("policy", "predictions")]))
        return (
            prediction[("policy", "predictions")],
            prediction[("value", "predictions")][0],
        )

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()

