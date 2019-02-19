from multiprocessing import Queue
from queue import Empty
from threading import Thread
from mathzero.model.features import (
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_COLUMNS,
)


class MathPredictor(object):
    """Provide a cached estimator implementation. This avoids re-creating the estimator for each
    prediction, which can cripple performance.

    Original implementation from: https://github.com/ElementAI/multithreaded-estimators :clap: :bow:
    """

    def __init__(self, estimator, verbose=False):
        self.verbose = verbose
        self.estimator = estimator
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.prediction_thread = Thread(target=self.predict_from_queue)
        self.prediction_thread.start()

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
            if self.verbose:
                print("Putting in output queue")
            self.output_queue.put(i)

    def predict(self, features):

        # Get predictions dictionary

        self.input_queue.put(features)
        predictions = self.output_queue.get()  # The latest predictions generator

        return predictions

    def queued_predict_input_fn(self):
        import tensorflow as tf

        output_types = {
            FEATURE_TOKEN_VALUES: tf.string,
            FEATURE_TOKEN_TYPES: tf.int64,
            FEATURE_NODE_COUNT: tf.int32,
            FEATURE_PROBLEM_TYPE: tf.int32,
        }
        dataset = tf.data.Dataset.from_generator(
            self.generate_from_queue, output_types=output_types
        )
        return dataset

    def destroy(self):
        self.input_queue.put(None)
        return self.prediction_thread.join()
