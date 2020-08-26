import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
from mathy.agent.action_selectors import predict_next
import numpy as np
import tensorflow as tf
from wasabi import msg

from ..env import MathyEnv
from ..envs.gym.mathy_gym_env import MathyGymEnv
from ..state import MathyEnvState, MathyObservation, observations_to_window
from ..teacher import Teacher
from . import action_selectors
from .config import AgentConfig
from .episode_memory import EpisodeMemory
from .model import (
    AgentLosses,
    AgentModel,
    call_model,
    compute_agent_loss,
    get_or_create_agent_model,
    save_model,
)
from .trfl import discrete_policy_entropy_loss, td_lambda
from .util import EpisodeLosses, record, truncate


class A3CWorker(threading.Thread):

    args: AgentConfig

    # <GLOBAL_VARS>
    global_episode = 0
    global_moving_average_reward = 0
    save_every_n_episodes = 250
    request_quit = False
    save_lock = threading.Lock()
    # </GLOBAL_VARS>

    envs: Dict[str, Any]

    losses: EpisodeLosses

    def __init__(
        self,
        args: AgentConfig,
        action_size: int,
        global_model: AgentModel,
        optimizer,
        greedy_epsilon: Union[float, List[float]],
        worker_idx: int,
        writer: tf.summary.SummaryWriter,
        teacher: Teacher,
        env_extra: dict,
    ):
        super(A3CWorker, self).__init__()
        self.args = args
        self.env_extra = env_extra
        self.greedy_epsilon = greedy_epsilon
        self.iteration = 0
        self.action_size = action_size
        self.global_model = global_model
        self.optimizer = optimizer
        self.worker_idx = worker_idx
        self.teacher = teacher
        self.losses = EpisodeLosses()

        with msg.loading(f"Worker {worker_idx} starting..."):
            first_env = self.teacher.get_env(self.worker_idx, self.iteration)
            self.writer = writer
            self.local_model = get_or_create_agent_model(
                config=args,
                predictions=self.action_size,
                env=gym.make(first_env, **self.env_extra).mathy,
            )
            self.last_model_write = -1
            self.last_histogram_write = -1
        display_e = self.greedy_epsilon
        if self.worker_idx == 0 and self.args.main_worker_use_epsilon is False:
            display_e = 0.0
        msg.good(f"Worker {worker_idx} started. (e={display_e:.3f})")

    @property
    def tb_prefix(self) -> str:
        if self.worker_idx == 0:
            return "agent"
        return f"workers/{self.worker_idx}"

    def run(self):
        if self.args.profile:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        episode_memory = EpisodeMemory(self.args.max_len)
        while (
            A3CWorker.global_episode < self.args.max_eps
            and A3CWorker.request_quit is False
        ):
            reward = self.run_episode(episode_memory)
            if (
                A3CWorker.global_episode
                > self.args.teacher_start_evaluations_at_episode
            ):
                win_pct = self.teacher.report_result(self.worker_idx, reward)
                if win_pct is not None:
                    with self.writer.as_default():
                        student = self.teacher.students[self.worker_idx]
                        step = self.global_model.optimizer.iterations
                        if self.worker_idx == 0:
                            tf.summary.scalar(
                                f"win_rate/{student.topic}", data=win_pct, step=step
                            )

            self.iteration += 1
            # TODO: Make this a subprocess? Python threads won't scale up well to
            #       many cores, I think.

        if self.args.profile:
            profile_name = f"worker_{self.worker_idx}.profile"
            profile_path = os.path.join(self.args.model_dir, profile_name)
            pr.disable()
            pr.dump_stats(profile_path)
            if self.args.verbose:
                print(f"PROFILER: saved {profile_path}")

    def run_episode(self, episode_memory: EpisodeMemory) -> float:
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)
        env: MathyGymEnv = gym.make(env_name, **self.env_extra)
        episode_memory.clear()
        self.ep_loss = 0
        ep_reward = 0.0
        ep_steps = 1
        time_count = 0
        done = False
        last_observation: MathyObservation = env.reset()
        last_action: Tuple[int, int] = (-1, -1)
        last_reward: float = 0.0
        while not done and A3CWorker.request_quit is False:
            if self.args.print_training and self.worker_idx == 0:
                env.render(last_action=last_action, last_reward=last_reward)
            window = episode_memory.to_window_observation(
                last_observation, window_size=self.args.prediction_window_size
            )
            action, value = predict_next(self.local_model, window.to_inputs())
            observation, reward, done, last_obs_info = env.step(action)
            ep_reward += reward
            episode_memory.store(
                observation=last_observation, action=action, reward=reward, value=value,
            )
            if time_count == self.args.update_gradients_every or done:
                if done and self.args.print_training and self.worker_idx == 0:
                    env.render(last_action=action, last_reward=last_reward)

                self.update_global_network(done, observation, episode_memory)
                self.maybe_write_histograms()
                time_count = 0
                if done:
                    self.finish_episode(
                        last_obs_info.get("win", False),
                        ep_reward,
                        ep_steps,
                        env.state,
                        env.mathy,
                    )

            ep_steps += 1
            time_count += 1
            last_observation = observation
            last_action = action
            last_reward = reward

            # If there are multiple workers, apply a worker sleep
            # to give the system some breathing room.
            if self.args.num_workers > 1:
                # The greedy worker sleeps for a shorter period of time
                sleep = self.args.worker_wait
                if self.worker_idx == 0:
                    sleep = max(sleep // 100, 0.005)
                # Workers wait between each step so that it's possible
                # to run more workers than there are CPUs available.
                time.sleep(sleep)
        return ep_reward

    def maybe_write_episode_summaries(
        self, episode_reward: float, episode_steps: int, last_state: MathyEnvState
    ):
        assert self.worker_idx == 0, "only write summaries for greedy worker"
        # Track metrics for all workers
        name = self.teacher.get_env(self.worker_idx, self.iteration)
        step = self.global_model.optimizer.iterations
        with self.writer.as_default():
            agent_state = last_state.agent
            steps = int(last_state.max_moves - agent_state.moves_remaining)
            rwd = truncate(episode_reward)
            p_text = f"{agent_state.history[0].raw} = {agent_state.history[-1].raw}"
            outcome = "SOLVED" if episode_reward > 0.0 else "FAILED"
            out_text = f"{outcome} [steps: {steps}, reward: {rwd}]: {p_text}"
            tf.summary.text(f"{name}/summary", data=out_text, step=step)

            # Track global model metrics
            tf.summary.scalar(
                f"agent/mean_episode_reward",
                data=A3CWorker.global_moving_average_reward,
                step=step,
            )

    def maybe_write_histograms(self) -> None:
        if self.worker_idx != 0:
            return
        step = self.global_model.optimizer.iterations.numpy()
        next_write = self.last_histogram_write + self.args.summary_interval
        if step >= next_write or self.last_histogram_write == -1:
            with self.writer.as_default():
                self.last_histogram_write = step
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.optimizer.iterations,
                    )

    def update_global_network(
        self, done: bool, observation: MathyObservation, episode_memory: EpisodeMemory,
    ):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            losses: AgentLosses = self.compute_loss(
                done=done,
                observation=observation,
                episode_memory=episode_memory,
                gamma=self.args.gamma,
            )
        self.losses.increment("loss", losses.total)
        self.losses.increment("fn_pi", losses.fn_policy)
        self.losses.increment("args_pi", losses.args_policy)
        self.losses.increment("v", losses.value)
        self.losses.increment("fn_h", losses.fn_entropy)
        self.losses.increment("args_h", losses.args_entropy)

        # Calculate local gradients
        grads = tape.gradient(losses.total, self.local_model.trainable_weights)
        # Push local gradients to global model

        zipped_gradients = zip(grads, self.global_model.trainable_weights)
        # Assert that we always have some gradient flow in each trainable var

        self.optimizer.apply_gradients(zipped_gradients)
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())

        if done:
            episode_memory.clear()
        else:
            episode_memory.clear_except_window(self.args.prediction_window_size)

    def finish_episode(
        self,
        is_win: bool,
        episode_reward: float,
        episode_steps: int,
        last_state: MathyEnvState,
        env: MathyEnv,
    ):
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)

        A3CWorker.global_moving_average_reward = record(
            A3CWorker.global_episode,
            is_win,
            episode_reward,
            self.worker_idx,
            A3CWorker.global_moving_average_reward,
            self.losses,
            episode_steps,
            env_name,
            env=env,
            state=last_state,
        )
        if self.worker_idx == 0:
            self.maybe_write_episode_summaries(
                episode_reward, episode_steps, last_state
            )
            step = self.global_model.optimizer.iterations.numpy()
            next_write = self.last_model_write + A3CWorker.save_every_n_episodes
            if step >= next_write or self.last_model_write == -1:
                self.last_model_write = step
                self.write_global_model()
        A3CWorker.global_episode += 1
        self.losses.reset()

    def write_global_model(self, increment_episode=True):
        with A3CWorker.save_lock:
            # Do this inside the lock so other threads can't also acquire the
            # lock in the time between when it's released and assigned outside
            # of the if conditional.
            model_path = os.path.join(self.args.model_dir, self.args.model_name)
            if increment_episode is True:
                A3CWorker.global_episode += 1
                save_model(self.global_model, model_path)

    def compute_loss(
        self,
        *,
        done: bool,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        gamma=0.99,
    ) -> AgentLosses:
        with self.writer.as_default():
            step = self.global_model.optimizer.iterations
            if done:
                bootstrap_value = 0.0  # terminal
            else:
                # Predict the reward using the local network
                _, _, values = call_model(
                    self.local_model,
                    observations_to_window(
                        [observation], self.args.max_len
                    ).to_inputs(),
                )
                # Select the last timestep
                bootstrap_value = float(tf.squeeze(values[-1]).numpy())

            losses, _ = compute_agent_loss(
                model=self.local_model,
                args=self.args,
                episode_memory=episode_memory,
                bootstrap_value=bootstrap_value,
                gamma=gamma,
            )
            prefix = self.tb_prefix
            tf.summary.scalar(
                f"losses/{prefix}/fn_policy_loss", data=losses.fn_policy, step=step
            )
            tf.summary.scalar(
                f"losses/{prefix}/args_policy_loss", data=losses.args_policy, step=step
            )
            tf.summary.scalar(
                f"losses/{prefix}/fn_entropy_loss", data=losses.fn_entropy, step=step
            )
            tf.summary.scalar(
                f"losses/{prefix}/args_entropy_loss",
                data=losses.args_entropy,
                step=step,
            )
            tf.summary.scalar(
                f"losses/{prefix}/value_loss", data=losses.value, step=step
            )
            tf.summary.scalar(
                f"settings/learning_rate", data=self.optimizer.lr(step), step=step
            )
            tf.summary.scalar(
                f"{self.tb_prefix}/total_loss", data=losses.total, step=step
            )

        return losses
