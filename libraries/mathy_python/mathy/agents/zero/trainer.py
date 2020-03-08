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

            batch_steps = int(len(examples) / self.args.batch_size)
            bar = Bar("Training Net", max=batch_steps)
            batch_idx = 0

            # self.session.run(tf.local_variables_initializer())
            while batch_idx < total_batches:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                (
                    text,
                    action,
                    reward,
                    discounted,
                    terminal,
                    observation,
                    pi_,
                    v,
                ) = list(zip(*[examples[i] for i in sample_ids]))
                pi = tf.keras.preprocessing.sequence.pad_sequences(pi_)
                inputs = observations_to_window(
                    [MathyObservation(*o) for o in observation]
                )
                with tf.GradientTape() as tape:
                    pi_loss, value_loss, total_loss = self.compute_loss(
                        gamma=self.args.gamma, inputs=inputs, target_pi=pi, target_v=v,
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
                bar.suffix = "({batch}/{size}) data: {data:.3f}s batch: {bt:.3f}s total: {total:} eta: {eta:} pi: {lpi:.4f} v: {lv:.3f}".format(
                    batch=batch_idx,
                    size=batch_steps,
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
        logits, values, _, _, _ = self.model(inputs.to_inputs())
        value_loss = tf.losses.mean_squared_error(
            target_v, tf.reshape(values, shape=[-1])
        )
        policy_logits = tf.reshape(logits, [batch_size, -1])
        policy_logits = policy_logits[:, : target_pi.shape[1]]
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_pi, logits=policy_logits
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
