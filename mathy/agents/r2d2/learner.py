import math
import os
import threading
import time
from multiprocessing import Process, Queue
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from colr import color

from ...core.expressions import MathTypeKeysMax
from ...features import FEATURE_FWD_VECTORS, calculate_grouping_control_signal
from ...state import (
    MathyBatchObservation,
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
    windows_to_batch,
)
from ...teacher import Student, Teacher, Topic
from ...util import GameRewards
from ..episode_memory import EpisodeMemory
from ..tensorflow.trfl import (
    discrete_policy_entropy_loss,
    discrete_policy_gradient_loss,
)
from ..base_config import BaseConfig
from ..experience import Experience, ExperienceFrame
from ..actor_critic_model import ActorCriticModel
from .util import MPClass, record, record_losses


class MathyLearner(MPClass):

    args: BaseConfig
    request_quit = False

    def __init__(
        self,
        args: BaseConfig,
        writer: tf.summary.SummaryWriter,
        command_queues: List[Queue],
        experience: Experience,
        **kwargs,
    ):
        super(MathyLearner, self).__init__()
        self.args = args
        self.writer = writer
        self.command_queues = command_queues
        self.experience = experience
        if self.args.verbose:
            print(f"Agent: {os.path.join(args.model_dir, args.model_name)}")
            print(f"Config: {self.args.dict()}")
        self.teacher = Teacher(
            topic_names=self.args.topics,
            num_students=self.args.num_workers,
            difficulty=self.args.difficulty,
        )
        self.env = gym.make(self.teacher.get_env(0, 0))
        self.action_size = self.env.action_space.n
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, "tensorboard")
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        self.model = ActorCriticModel(
            args=args, predictions=self.action_size, optimizer=self.optimizer
        )
        self.obs_converter = EpisodeMemory(self.experience)
        # Initialize the model with a random observation
        self.model.maybe_load(self.env.initial_state(), do_init=True)
        self.update_actors_in = self.args.update_freq

    def choose_action(self, state: MathyEnvState, greedy=False):
        obs = state.to_input_features(self.env.action_space.mask, return_batch=True)
        if greedy is True:
            policy, value, masked_policy = self.model.call_masked(obs)
            policy = tf.nn.softmax(masked_policy)
            action = np.argmax(policy)
        else:
            probs, value = self.model.predict_next(obs)
            action = np.random.choice(len(probs), p=probs)
        return action

    def run(self):
        model = self.model
        model.maybe_load()

        try:
            while MathyLearner.request_quit is False:
                if not self.experience.is_full():
                    time.sleep(1.5)
                    continue
                self.train_batch_samples()
                self.maybe_update_actor_models()

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            pass

    def maybe_update_actor_models(self):
        """periodically send commands to each actor telling them to reload their
        model from the latest weights."""
        if self.update_actors_in == 0:
            self.model.save()
            [q.put("load_model") for q in self.command_queues]
            self.update_actors_in = self.args.actor_update_from_learner_every_n
        else:
            self.update_actors_in -= 1

    def maybe_write_histograms(self):
        if self.worker_idx != 0:
            return
        # The global step is incremented when the optimizer is applied, so check
        # and print summary data here.
        summary_interval = 100
        with self.writer.as_default():
            with tf.summary.record_if(
                lambda: tf.math.equal(self.model.global_step % summary_interval, 0)
            ):
                for var in self.model.trainable_variables:
                    tf.summary.histogram(var.name, var, step=self.model.global_step)

    def train_batch_samples(self):
        if not self.experience.is_full():
            return
        obs_windows: List[MathyWindowObservation] = []

        batch_frames: List[List[ExperienceFrame]] = []
        batch_rnn_states: List[List[float]] = []
        batch_discounted_rewards: List[float] = []
        batch_action_labels: List[int] = []

        for i in range(self.args.batch_size):
            window: List[ExperienceFrame] = self.experience.sample_sequence(3, self.env)
            batch_frames.append(window)
            states: List[MathyObservation] = []
            action_labels: List[int] = []
            rnn_states: List[List[float]] = [[], []]
            discounted_rewards: List[float] = []
            for frame in window:
                states.append(frame.state)
                rnn_states[0].append(frame.rnn_state[0][-1])
                rnn_states[1].append(frame.rnn_state[1][-1])
                discounted_rewards.append(frame.discounted)
            action_labels.append(window[-1].action)
            rnn_states = [
                tf.convert_to_tensor(rnn_states[0], dtype=tf.float32),
                tf.convert_to_tensor(rnn_states[1], dtype=tf.float32),
            ]
            discounted_rewards = tf.convert_to_tensor(
                discounted_rewards, dtype=tf.float32
            )
            action_labels = tf.convert_to_tensor(action_labels)

            obs_windows.append(observations_to_window(states))
            batch_rnn_states.append(rnn_states)
            batch_discounted_rewards.append(discounted_rewards)
            batch_action_labels.append(action_labels)
        batch_observations: MathyBatchObservation = windows_to_batch(obs_windows)
        return self.update_loss(
            batch_frames,
            batch_observations,
            batch_rnn_states,
            batch_discounted_rewards,
            batch_action_labels,
        )

    def update_loss(
        self,
        batch_frames: List[List[ExperienceFrame]],
        batch_observations: MathyBatchObservation,
        batch_rnn_states: List[List[List[float]]],
        batch_discounted_rewards: List[List[float]],
        batch_action_labels: List[List[int]],
    ):
        # Calculate gradient by tracking the variables involved in computing the
        # loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            loss_tuple = self.compute_loss(
                episode_memory=self.obs_converter,
                batch_frames=batch_frames,
                batch_observations=batch_observations,
                rnn_states=batch_rnn_states,
                discounted_rewards=batch_discounted_rewards,
                action_labels=batch_action_labels,
            )
            pi_loss, value_loss, entropy_loss, aux_losses, total_loss = loss_tuple
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.iterations = self.model.global_step
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights), global_step=self.model.global_step
        )
        for k in aux_losses.keys():
            aux_losses[k] = aux_losses[k].numpy()
        step = self.model.global_step.numpy()
        # if step % 10 == 0:
        record_losses(step, pi_loss, value_loss, entropy_loss, aux_losses, total_loss)
        return loss_tuple

    def write_global_model(self):
        self.model.save()

    def compute_policy_value_loss(
        self,
        batch_observations: MathyBatchObservation,
        rnn_states: List[tf.Tensor],
        discounted_rewards: List[tf.Tensor],
        action_labels: List[tf.Tensor],
    ):
        batch_size = len(discounted_rewards)
        step = self.model.global_step
        logits, values, trimmed_logits = self.model(
            batch_observations, initial_state=rnn_states
        )
        n_step_size = logits.shape[1]
        logits = tf.reshape(logits, [batch_size, -1])
        masked_flat = tf.reshape(trimmed_logits, [batch_size, -1])

        # Calculate entropy and policy loss
        h_loss = discrete_policy_entropy_loss(logits)
        # pi_loss = discrete_policy_gradient_loss(logits, action_labels, reward_values)

        # Advantage is the difference between the final calculated discounted
        # rewards, and the current Value function prediction of the rewards
        last_t_discounts = tf.convert_to_tensor(discounted_rewards)[:, -1]
        advantage = last_t_discounts - values

        # Value loss is the squared error (advantage is the prediction error for value)
        value_loss = advantage ** 2

        # Policy Loss

        # We calculate policy loss from the masked logits to keep
        # the error from exploding when irrelevant (masked) logits
        # have large values. Because we apply a mask for all operations
        # we don't care what those logits are, unless they're part of
        # the mask.
        labels = tf.squeeze(tf.convert_to_tensor(action_labels), axis=1)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        policy_loss *= tf.stop_gradient(advantage)

        value_loss *= 0.5
        total_loss = tf.reduce_mean(value_loss + policy_loss + h_loss.loss)
        tf.summary.scalar(
            f"losses/policy_loss", data=tf.reduce_mean(policy_loss), step=step
        )
        tf.summary.scalar(
            f"losses/value_loss", data=tf.reduce_mean(value_loss), step=step
        )
        tf.summary.scalar(
            f"losses/entropy_loss", data=tf.reduce_mean(h_loss.loss), step=step
        )
        tf.summary.scalar(f"advantage", data=tf.reduce_sum(advantage), step=step)
        tf.summary.scalar(
            f"entropy", data=tf.reduce_mean(h_loss.extra.entropy), step=step
        )

        return (
            tf.reduce_mean(policy_loss),
            tf.reduce_mean(value_loss),
            tf.reduce_mean(h_loss.loss),
            total_loss,
        )

    def compute_grouping_change_loss(self, batch_frames: List[List[ExperienceFrame]]):
        change_signals = []
        for window in batch_frames:
            change_signals.append([frame.grouping_change for frame in window])
        loss = tf.reduce_mean(tf.convert_to_tensor(change_signals))
        return loss

    def compute_reward_prediction_loss(
        self, batch_frames: List[List[ExperienceFrame]], episode_memory: EpisodeMemory
    ):
        if not self.experience.is_full():
            return tf.constant(0.0)
        rp_inputs, rp_labels = self.rp_samples()
        rp_losses = []
        rp_outputs = self.model.predict_next_reward(rp_inputs)
        rp_losses.append(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=rp_outputs, labels=rp_labels
            )
        )
        return tf.reduce_mean(tf.convert_to_tensor(rp_losses))

    def compute_value_replay_loss(
        self,
        samples: List[ExperienceFrame],
        episode_memory: EpisodeMemory,
        discounted_rewards: List[float],
    ):
        vr_losses = []
        for i, frame in enumerate(samples):
            states = [frame.state for frame in samples]
            sample_features = episode_memory.to_features(states)
            vr_values = self.model.predict_value_replays(sample_features)
            advantage = discounted_rewards[i] - vr_values
            # Value loss
            value_loss = advantage ** 2
            vr_losses.append(value_loss)

        return tf.reduce_mean(tf.convert_to_tensor(vr_losses))

    def compute_loss(
        self,
        episode_memory: EpisodeMemory,
        batch_frames: List[List[ExperienceFrame]],
        batch_observations: MathyBatchObservation,
        rnn_states: List[tf.Tensor],
        discounted_rewards: tf.Tensor,
        action_labels: tf.Tensor,
    ):
        with self.writer.as_default():
            step = self.model.global_step
            loss_tuple = self.compute_policy_value_loss(
                batch_observations, rnn_states, discounted_rewards, action_labels
            )
            pi_loss, v_loss, h_loss, total_loss = loss_tuple
            aux_losses = {}
            aux_weight = 0.2
            if self.args.use_grouping_control:
                gc_loss = self.compute_grouping_change_loss(batch_frames)
                # gc_loss *= aux_weight
                total_loss += gc_loss
                aux_losses["gc"] = gc_loss
            if self.args.use_reward_prediction:
                rp_loss = self.compute_reward_prediction_loss(
                    batch_frames, episode_memory
                )
                rp_loss *= aux_weight
                total_loss += rp_loss
                aux_losses["rp"] = rp_loss
            if self.args.use_value_replay:
                vr_loss = self.compute_value_replay_loss(
                    batch_frames, episode_memory, discounted_rewards
                )
                vr_loss *= aux_weight
                total_loss += vr_loss
                aux_losses["vr"] = vr_loss
            for key in aux_losses.keys():
                tf.summary.scalar(f"losses/{key}_loss", data=aux_losses[key], step=step)

            tf.summary.scalar(f"losses/total_loss", data=total_loss, step=step)

        return pi_loss, v_loss, h_loss, aux_losses, total_loss

    # Auxiliary tasks

    def rp_samples(self, max_samples=2) -> Tuple[MathyBatchObservation, List[float]]:
        output: MathyBatchObservation = MathyBatchObservation([], [], [], [[], []])
        rewards: List[float] = []
        if self.experience.is_full() is False:
            return output, rewards
        windows: List[MathyWindowObservation] = []
        for i in range(max_samples):
            frames = self.experience.sample_rp_sequence()
            # 4 frames
            states = [frame.state for frame in frames[:-1]]
            target_reward = frames[-1].reward
            if math.isclose(target_reward, GameRewards.TIMESTEP):
                sample_label = 0  # zero
            elif target_reward > 0:
                sample_label = 1  # positive
            else:
                sample_label = 2  # negative
            windows.append(observations_to_window(states))
            rewards.append(sample_label)
        return windows_to_batch(windows), rewards

    def get_rp_inputs_with_labels(
        self, episode_buffer: EpisodeMemory
    ) -> Tuple[List[Any], List[Any]]:
        # [Reward prediction]
        rp_experience_frames: List[
            ExperienceFrame
        ] = self.agent_experience.sample_rp_sequence()
        # 4 frames
        states = [frame.state for frame in rp_experience_frames[:-1]]
        batch_rp_si = episode_buffer.to_features(states)
        batch_rp_c = []

        # one hot vector for target reward
        r = rp_experience_frames[3].reward
        rp_c = [0.0, 0.0, 0.0]
        if math.isclose(r, GameRewards.TIMESTEP):
            rp_c[0] = 1.0  # zero
        elif r > 0:
            rp_c[1] = 1.0  # positive
        else:
            rp_c[2] = 1.0  # negative
        batch_rp_c.append(rp_c)
        return batch_rp_si, batch_rp_c
