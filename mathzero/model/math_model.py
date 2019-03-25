import sys
import collections
import os
from colr import color
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
    FEATURE_LAST_TOKEN_VALUES,
    FEATURE_LAST_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_PROBLEM_TYPE,
)


class NetConfig:
    def __init__(
        self, lr=0.01, dropout=0.2, epochs=1, batch_size=256, log_frequency=250
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
        root_dir,
        all_memory=False,
        dev_mode=False,
        init_model_dir=None,
        init_model_overwrite=False,
        embeddings_dimensions=256,
        long_term_size=640,
        is_eval_model=False,
    ):
        import tensorflow as tf

        self.is_eval_model = is_eval_model
        self.init_model_overwrite = init_model_overwrite
        self.long_term_size = long_term_size
        self.root_dir = root_dir
        if not is_eval_model:
            self.model_dir = os.path.join(self.root_dir, "train")
        else:
            self.model_dir = os.path.join(self.root_dir, "eval")
        self.init_model_dir = init_model_dir
        if (
            self.init_model_dir is not None
            and Path(self.model_dir).is_dir()
            and self.init_model_overwrite is not True
        ):
            print(
                "-- skipping trainable variables transfer from model: "
                "(checkpoint exists and overwrite is false)"
            )
            self.init_model_dir = None
        if self.init_model_dir is not None:
            print(
                "-- transferring trainable variables to blank model from: {}".format(
                    self.init_model_dir
                )
            )
        self.embedding_dimensions = embeddings_dimensions
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
        self.f_last_token_types_sequence = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_hash_bucket(
                key=FEATURE_LAST_TOKEN_TYPES, hash_bucket_size=24, dtype=tf.int8
            ),
            dimension=32,
        )
        self.f_last_token_values_sequence = tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_hash_bucket(
                key=FEATURE_LAST_TOKEN_VALUES, hash_bucket_size=128, dtype=tf.string
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
        print(
            color(
                "-- init math model in: {}\ninit model dir: {}".format(
                    self.model_dir, self.init_model_dir
                ),
                fore="blue",
                style="bright",
            )
        )
        self.network = tf.estimator.Estimator(
            warm_start_from=self.init_model_dir,
            config=estimator_config,
            model_fn=math_estimator,
            model_dir=self.model_dir,
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

    def train(self, short_term_examples, long_term_examples, train_all=False):
        """examples: list of examples in JSON format"""
        from .math_hooks import EpochTrainerHook
        import tensorflow as tf
        from .math_dataset import make_training_input_fn

        # Reflection capacity (how many observations should we train on in this meditation?)
        if train_all is True:
            max_examples = self.long_term_size
        else:
            max_examples = len(short_term_examples) + len(long_term_examples)

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
        # Shuffle all training examples
        random.shuffle(examples)
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

    def train_one(self, example):
        import tensorflow as tf
        from .math_dataset import make_training_input_fn

        start = time.time()
        self.network.train(steps=1, input_fn=make_training_input_fn([example], 1))
        print("train_one : {0:03f}".format(time.time() - start))
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

    def encode(self, env_state: MathEnvironmentState):
        """Encode the environment state into an embedding tensor that can be consumed by RL
        algorithms that demand a single tensor input"""
        input_features = env_state.to_input_features(return_batch=True)
        # start = time.time()
        prediction = self._worker.predict(input_features)
        # print("predict : {0:03f}".format(time.time() - start))
        # print("distribution is : {}".format(prediction[("policy", "predictions")]))
        # TODO: want to return the embeddings here. Try policy since we don't have time to fix the
        # export of the embeddings tensor in our pretrained model.
        return prediction[("policy", "predictions")]

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()

