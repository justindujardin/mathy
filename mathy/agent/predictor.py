from multiprocessing import Queue
from queue import Empty
from threading import Thread

import tensorflow as tf

from mathy.agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
)


class MathPredictor(object):
    """Provide a threaded estimator implementation that avoids re-creating the TF device for each
    prediction. Device creation is expensive if you do it for every prediction, and you are 
    doing a ton of predictions. For example the original Mathy model required about ~0.5 seconds 
    for a prediction, and that dropped to ~0.01 after using this cached predictor.

    Original implementation from: https://github.com/ElementAI/multithreaded-estimators
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.worker_thread = None

    def start(self):
        """Start a worker thread to predict with a shared device. Device creation can
        be expensive with GPUs so this speeds things up considerably in that use-case.
        """
        if self.worker_thread is not None:
            raise ValueError("thread is already started")
        worker_fn = self.predict_from_queue
        self.worker_thread = Thread(target=worker_fn)
        self.worker_thread.start()

    def stop(self):
        """Stop the current worker thread"""
        if self.worker_thread is None:
            raise ValueError("thread is already stopped")
        thread = self.worker_thread
        self.input_queue.put(None)
        self.worker_thread = None
        thread.join()

    def predict(self, features):
        """Predict from features for a given single input"""
        if self.worker_thread is None:
            raise ValueError("No thread started, so the prediction will never return")
        self.input_queue.put(features)
        predictions = self.output_queue.get()
        return predictions

    def generate_from_queue(self):
        while True:
            try:
                result = self.input_queue.get(timeout=1)
            except Empty:
                continue
            except EOFError:
                return
            if result is None:
                return
            yield result

    def predict_from_queue(self):
        for i in self.estimator.predict(input_fn=self.queued_predict_input_fn):
            self.output_queue.put(i)

    def queued_predict_input_fn(self):

        output_types = {
            FEATURE_FWD_VECTORS: tf.int64,
            FEATURE_BWD_VECTORS: tf.int64,
            FEATURE_LAST_FWD_VECTORS: tf.int64,
            FEATURE_LAST_BWD_VECTORS: tf.int64,
            FEATURE_LAST_RULE: tf.int64,
            FEATURE_NODE_COUNT: tf.int64,
            FEATURE_MOVE_COUNTER: tf.int64,
            FEATURE_MOVES_REMAINING: tf.int64,
            FEATURE_PROBLEM_TYPE: tf.int64,
        }
        dataset = tf.data.Dataset.from_generator(
            self.generate_from_queue, output_types=output_types
        )
        return dataset
