import tensorflow as tf
import logging
import time
from datetime import datetime


# This is defined inside the class because we don't import tf at
# the module level. Multi-process self-play explodes if tf is
# imported at the top of the file.
class TrainingLoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, log_frequency, batch_size):
        self.log_frequency = log_frequency
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
        examples_per_sec = self.log_frequency * self.batch_size / duration
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
