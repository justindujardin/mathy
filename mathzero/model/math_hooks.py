import tensorflow as tf
import logging
import time
import sys
from datetime import datetime


class TrainingLoggerHook(tf.train.SessionRunHook):
    """Log training progress to the console, including pi and value losses"""

    def __init__(self, log_frequency, batch_size):
        self.batch_size = batch_size

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(tf.get_collection("mt"))

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self._start_time
        self._start_time = current_time
        loss_pi, loss_v = run_values.results
        examples_per_sec = self.batch_size / duration
        sec_per_batch = duration
        template = "%s: step %d, loss_pi = %.2f, loss_v = %.2f (%.1f examples/sec; %.3f sec/batch)"
        args = (
            datetime.now(),
            self._step,
            loss_pi,
            loss_v,
            examples_per_sec,
            sec_per_batch,
        )
        print(template % args)
        sys.stdout.flush()
