import sys
import time
from datetime import datetime

from tensorflow.python.training import training
from tensorflow_estimator.python.estimator.hooks.session_run_hook import SessionRunHook

from ..features import (
    TENSOR_KEY_GROUP_PREDICT,
    TENSOR_KEY_GROUPING_CTRL,
    TENSOR_KEY_NODE_CTRL,
    TENSOR_KEY_PI,
    TENSOR_KEY_REWARD_PREDICT,
    TENSOR_KEY_VALUE,
)


class EpochTrainerHook(SessionRunHook):
    """Train for a given number of epochs, logging training progress to the console"""

    def __init__(
        self,
        epochs,
        examples_count,
        batch_size,
        node_ctrl=True,
        grouping_ctrl=True,
        group_prediction=True,
        reward_prediction=True,
    ):
        self.epochs = epochs
        self.examples_count = examples_count
        self.batch_size = batch_size
        self.node_ctrl = node_ctrl
        self.grouping_ctrl = grouping_ctrl
        self.group_prediction = group_prediction
        self.reward_prediction = reward_prediction

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):

        self._step += 1

        # These are the tensors from our multi-head predictions
        fetch_tensors = {
            TENSOR_KEY_PI: "policy/weighted_loss/value:0",
            TENSOR_KEY_VALUE: "value/weighted_loss/value:0",
        }
        if self.node_ctrl is True:
            fetch_tensors[TENSOR_KEY_NODE_CTRL] = "node_ctrl/weighted_loss/value:0"
        if self.grouping_ctrl is True:
            fetch_tensors[
                TENSOR_KEY_GROUPING_CTRL
            ] = "grouping_ctrl/weighted_loss/value:0"
        if self.grouping_ctrl is True:
            fetch_tensors[
                TENSOR_KEY_GROUP_PREDICT
            ] = "group_prediction/weighted_loss/value:0"
        if self.grouping_ctrl is True:
            fetch_tensors[
                TENSOR_KEY_REWARD_PREDICT
            ] = "reward_prediction/weighted_loss/value:0"

        return training.SessionRunArgs(fetch_tensors)

    def after_run(self, run_context, run_values):
        def truncate(value):
            return float("%.3f" % (float(value)))

        def maybe_value(value):
            return "%.3f" % value if value != 0.0 else "n/a"

        steps_per_epoch = max(int(self.examples_count / self.batch_size), 1)
        total_steps = self.epochs * steps_per_epoch
        if self._step % steps_per_epoch != 0 and self._step < total_steps:
            return

        if self._step >= total_steps:
            return run_context.request_stop()

        current_epoch = int(self._step / steps_per_epoch) + 1
        current_time = time.time()
        duration = current_time - self._start_time
        self._start_time = current_time
        loss_pi = truncate(run_values.results[TENSOR_KEY_PI])
        loss_v = truncate(run_values.results[TENSOR_KEY_VALUE])
        loss_nctrl = truncate(run_values.results.get(TENSOR_KEY_NODE_CTRL, 0.0))
        loss_gctrl = truncate(run_values.results.get(TENSOR_KEY_GROUPING_CTRL, 0.0))
        loss_gpred = truncate(run_values.results.get(TENSOR_KEY_GROUP_PREDICT, 0.0))
        loss_rpred = truncate(run_values.results.get(TENSOR_KEY_REWARD_PREDICT, 0.0))
        loss = loss_pi + loss_v + loss_nctrl + loss_gctrl + loss_gpred + loss_rpred
        examples_per_sec = steps_per_epoch * (self.batch_size / duration)
        sec_per_batch = duration
        template_args = (datetime.now().time(), current_epoch, loss, loss_pi, loss_v)
        template = "%s: epoch %d, total = %.3f pi = %.3f v = %.3f" % template_args
        aux_args = (
            maybe_value(loss_nctrl),
            maybe_value(loss_gctrl),
            maybe_value(loss_gpred),
            maybe_value(loss_rpred),
        )
        aux_template = "[nc = %s, gc = %s, gp = %s, rp = %s]" % aux_args
        timing_args = (examples_per_sec, sec_per_batch)
        timing_template = "(%.1fex/s; %.3fs/batch)" % timing_args
        output = f"{template} {aux_template} {timing_template}"
        print(output)
        sys.stdout.flush()
