from multiprocessing import Queue
from queue import Empty
from threading import Thread
from mathzero.model.features import (
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_MOVE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_COLUMNS,
)


class MathPredictor(object):
    """Provide a cached estimator implementation. This avoids re-creating the device for each
    prediction. Device creation can cripple performance for GPU workflows where it 
    is an expensive operation.

    Original implementation from: https://github.com/ElementAI/multithreaded-estimators
    """

    def __init__(self, estimator, args):
        self.args = args
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
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """

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
        """ Adds a prediction from the model to the output_queue.
        This lives within our 'prediction thread'.
        Note: estimators accept generators as inputs and return generators as output.
        Here, we are iterating through the output generator, which will be 
        populated in lock-step with the input generator.
        """
        for i in self.estimator.predict(input_fn=self.queued_predict_input_fn):
            self.output_queue.put(i)

    def queued_predict_input_fn(self):
        import tensorflow as tf

        output_types = {
            FEATURE_TOKEN_VALUES: tf.string,
            FEATURE_TOKEN_TYPES: tf.int64,
            FEATURE_NODE_COUNT: tf.int32,
            FEATURE_MOVE_COUNT: tf.int32,
            FEATURE_PROBLEM_TYPE: tf.int32,
        }
        dataset = tf.data.Dataset.from_generator(
            self.generate_from_queue, output_types=output_types
        )
        return dataset

