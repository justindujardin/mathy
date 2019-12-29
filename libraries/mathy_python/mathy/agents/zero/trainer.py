import math
import os
import queue
import threading
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np

from ...envs.gym.mathy_gym_env import MathyGymEnv
from ...util import calculate_grouping_control_signal
from ...state import (
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from ...teacher import Teacher
from ...util import discount
from .. import action_selectors
from ..episode_memory import EpisodeMemory
from ..policy_value_model import PolicyValueModel
from ..trfl import discrete_policy_entropy_loss, td_lambda
from .config import SelfPlayConfig
from .lib.average_meter import AverageMeter
from .lib.progress.bar import Bar


class SelfPlayTrainer:

    args: SelfPlayConfig

    def __init__(
        self, args: SelfPlayConfig, model: PolicyValueModel, action_size: int,
    ):
        super(SelfPlayTrainer, self).__init__()
        import tensorflow as tf

        self.args = args
        self.model = model
        self.iteration = 0
        self.action_size = action_size
        self.writer = None
        self.last_histogram_write = -1
        if self.args.use_grouping_control:
            raise NotImplementedError(
                "Grouping Control signal is not implemented for the Zero agent."
                " Support shouldn't be very difficult to add by looking at the"
                " compute_policy_value_loss in the A3C agent."
            )

    @property
    def tb_prefix(self) -> str:
        return "agent"

    def train(self, examples, model):

        total_batches = int(len(examples) / self.args.batch_size)
        if total_batches == 0:
            return False

        import tensorflow as tf

        print(
            "Training neural net for ({}) epochs with ({}) examples...".format(
                self.args.epochs, len(examples)
            )
        )

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar("Training Net", max=self.args.batch_size)
            batch_idx = 0

            # self.session.run(tf.local_variables_initializer())
            while batch_idx < total_batches:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                sequences = [examples[i] for i in sample_ids]
                for seq in sequences:
                    samp = list(zip(*seq))
                    (
                        text,
                        action,
                        reward,
                        discounted,
                        terminal,
                        observation,
                        pi,
                        v,
                    ) = samp
                    pi = tf.keras.preprocessing.sequence.pad_sequences(pi)
                    inputs = observations_to_window(
                        [MathyObservation(*o) for o in observation]
                    )
                    with tf.GradientTape() as tape:
                        pi_loss, value_loss, total_loss = self.compute_loss(
                            gamma=self.args.gamma,
                            inputs=inputs,
                            target_pi=pi,
                            target_v=v,
                        )
                    grads = tape.gradient(total_loss, self.model.trainable_weights)
                    zipped_gradients = zip(grads, self.model.trainable_weights)
                    self.model.optimizer.apply_gradients(zipped_gradients)

                    # measure data loading time
                    data_time.update(time.time() - end)

                    pi_losses.update(pi_loss, len(text))
                    v_losses.update(value_loss, len(text))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    batch_idx += 1

                    # plot progress
                    bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}".format(
                        batch=batch_idx,
                        size=self.args.batch_size,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        lpi=pi_losses.avg,
                        lv=v_losses.avg,
                    )
                    bar.next()
            bar.finish()
        return True

    def compute_policy_value_loss(
        self, inputs: MathyWindowObservation, target_pi: Any, target_v: Any, gamma=0.99,
    ):
        import tensorflow as tf

        batch_size = len(inputs.nodes)
        step = self.model.optimizer.iterations
        logits, values, trimmed_logits = self.model(
            inputs.to_inputs(), apply_mask=False
        )
        value_loss = tf.losses.mean_squared_error(
            target_v, tf.reshape(values, shape=[-1])
        )
        policy_logits = tf.reshape(trimmed_logits, [batch_size, -1])
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(target_pi, policy_logits.shape), logits=policy_logits
        )
        policy_loss = tf.reduce_mean(policy_loss)
        total_loss = value_loss + policy_loss
        prefix = self.tb_prefix
        tf.summary.scalar(f"{prefix}/policy_loss", data=policy_loss, step=step)
        tf.summary.scalar(f"{prefix}/value_loss", data=value_loss, step=step)

        return (policy_loss, value_loss, total_loss)

    def compute_loss(
        self, inputs: MathyWindowObservation, target_pi: Any, target_v: Any, gamma=0.99,
    ):
        loss_tuple = self.compute_policy_value_loss(inputs, target_pi, target_v)
        pi_loss, v_loss, total_loss = loss_tuple
        return pi_loss, v_loss, total_loss
