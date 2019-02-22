import tensorflow as tf
import logging
import time
import sys
from datetime import datetime
from math import isclose


class TrainingLoggerHook(tf.train.SessionRunHook):
    """Log training progress to the console, including pi and value losses"""

    def __init__(self, batch_size, log_every_n=100):
        self.batch_size = batch_size
        self.log_every_n = log_every_n

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(tf.get_collection("mt"))

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self._start_time
        if self._step % self.log_every_n != 0:
            return
        self._start_time = current_time
        loss_pi, loss_v = run_values.results
        examples_per_sec = self.log_every_n * self.batch_size / duration
        sec_per_batch = duration
        template = "%s: step %d, loss = %.3f loss_pi = %.3f, loss_v = %.3f (%.1f examples/sec; %.3f sec/batch)"
        args = (
            datetime.now(),
            self._step,
            loss_pi + loss_v,
            loss_pi,
            loss_v,
            examples_per_sec,
            sec_per_batch,
        )
        print(template % args)
        sys.stdout.flush()


class TrainingEarlyStopHook(tf.train.SessionRunHook):
    def __init__(
        self, watch_pi=True, watch_value=True, stop_after_n_steps_without_change=250
    ):
        self.num_steps = stop_after_n_steps_without_change
        self.steps_without_change = 0
        self.watch_pi = watch_pi
        self.watch_value = watch_value
        self.last_loss_pi = 0
        self.last_loss_v = 0

    def after_run(self, run_context, run_values):
        def truncate(value):
            return float("%.3f" % (float(value)))

        loss_pi, loss_v = run_values.results
        loss_pi = truncate(loss_pi)
        loss_v = truncate(loss_v)
        self.steps_without_change = self.steps_without_change + 1
        if (
            self.watch_pi is True
            and loss_pi < self.last_loss_pi
            and not isclose(loss_pi, self.last_loss_pi)
        ):
            self.steps_without_change = 0
        elif (
            self.watch_value is True
            and loss_v < self.last_loss_v
            and not isclose(loss_v, self.last_loss_v)
        ):
            self.steps_without_change = 0
        self.last_loss_v = loss_v
        self.last_loss_pi = loss_pi
        if self.steps_without_change >= self.num_steps:
            print("STOPPING because monitored metrics stopped decreasing.")
            run_context.request_stop()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(tf.get_collection("mt"))

