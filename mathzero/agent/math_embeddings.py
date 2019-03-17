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
    def __init__(self, network):
        import tensorflow as tf
        self.args = NetConfig()

        self.network = network
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

