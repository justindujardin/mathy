import tensorflow as tf
import logging
import time
import sys
from datetime import datetime
from math import isclose
from tensorflow_estimator.python.estimator.hooks.session_run_hook import SessionRunHook

# These are the tensors from our multi-head predictions. We fetch their current values
# in the below hooks
MATH_OUTPUT_TENSORS = {
    "policy": "policy/weighted_loss/value:0",
    "value": "value/weighted_loss/value:0",
}


class EpochTrainerHook(SessionRunHook):
    """Train for a given number of epochs, logging training progress to the console"""

    def __init__(self, epochs, examples_count, batch_size):
        self.epochs = epochs
        self.examples_count = examples_count
        self.batch_size = batch_size

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        global MATH_OUTPUT_TENSORS
        from tensorflow.python.training import training
        import tensorflow as tf

        self._step += 1

        return training.SessionRunArgs(MATH_OUTPUT_TENSORS)

    def after_run(self, run_context, run_values):
        import tensorflow as tf

        def truncate(value):
            return float("%.3f" % (float(value)))

        steps_per_epoch = max(int(self.examples_count / self.batch_size), 1)
        total_steps = self.epochs * steps_per_epoch
        if self._step % steps_per_epoch != 0:
            return
        current_epoch = int(self._step / steps_per_epoch) + 1
        current_time = time.time()
        duration = current_time - self._start_time
        self._start_time = current_time
        loss_pi = truncate(run_values.results["policy"])
        loss_v = truncate(run_values.results["value"])
        examples_per_sec = steps_per_epoch * (self.batch_size / duration)
        sec_per_batch = duration
        template = "%s: Epoch %d, loss = %.3f loss_pi = %.3f, loss_v = %.3f (%.1f examples/sec; %.3f sec/batch)"
        args = (
            datetime.now(),
            current_epoch,
            loss_pi + loss_v,
            loss_pi,
            loss_v,
            examples_per_sec,
            sec_per_batch,
        )
        print(template % args)
        sys.stdout.flush()
        # Stop after the last epoch
        if self._step >= total_steps:
            print("-- STOPPING AFTER LAST EPOCH")
            return run_context.request_stop()


class TrainingEarlyStopHook(SessionRunHook):
    def __init__(
        self,
        watch_pi=True,
        watch_value=False,
        stop_after_n=50,
        min_steps=500,
    ):
        self.num_steps = stop_after_n
        self.min_steps = min_steps
        self.steps_without_change = 0
        self.watch_pi = watch_pi
        self.watch_value = watch_value
        self.last_loss_pi = 0
        self.last_loss_v = 0

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        global MATH_OUTPUT_TENSORS
        from tensorflow.python.training import training
        import tensorflow as tf

        self._step += 1

        return training.SessionRunArgs(MATH_OUTPUT_TENSORS)

    def after_run(self, run_context, run_values):
        import tensorflow as tf

        def truncate(value):
            return float("%.3f" % (float(value)))

        loss_pi = truncate(run_values.results["policy"])
        loss_v = truncate(run_values.results["value"])
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
        if self.steps_without_change >= self.num_steps and self._step > self.min_steps:
            print(
                "STOPPING because monitored metrics stopped decreasing, and min-steps exceeded."
            )
            run_context.request_stop()

    def before_run(self, run_context):
        global MATH_OUTPUT_TENSORS
        from tensorflow.python.training import training

        return training.SessionRunArgs(MATH_OUTPUT_TENSORS)

