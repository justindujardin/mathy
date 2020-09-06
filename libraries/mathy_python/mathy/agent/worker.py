import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np
import tensorflow as tf
from wasabi import msg

from ..env import MathyEnv
from ..envs.gym.mathy_gym_env import MathyGymEnv
from ..state import MathyEnvState, MathyObservation, MathyWindowObservation
from ..teacher import Teacher
from ..types import ActionList, ActionType, RewardList, ValueList
from .config import AgentConfig
from .episode_memory import EpisodeMemory
from .model import (
    AgentLosses,
    AgentModel,
    call_model,
    compute_agent_loss,
    get_or_create_agent_model,
    predict_action_value,
    save_model,
)
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
        worker_idx: int,
        writer: tf.summary.SummaryWriter,
        teacher: Teacher,
        env_extra: dict,
    ):
        super(A3CWorker, self).__init__()
        self.args = args
        self.env_extra = env_extra
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
        msg.good(f"Worker {worker_idx} started.")

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

        episode_memory = EpisodeMemory(
            self.args.max_len, self.args.prediction_window_size
        )
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
        last_action: ActionType = (-1, -1)
        last_reward: float = 0.0

        ep_rules: List[int] = []
        ep_nodes: List[int] = []
        ep_rewards: List[float] = []
        ep_value_estimates: List[float] = []

        # TODO: Track episode stats for
        #  action mean
        #  action stddev
        #  node mean
        #  node stddev
        #  observation

        while not done and A3CWorker.request_quit is False:
            if self.args.print_training and self.worker_idx == 0:
                env.render(last_action=last_action, last_reward=last_reward)
            window = episode_memory.to_window_observation(last_observation)
            action, value = predict_action_value(self.local_model, window.to_inputs())
            observation, reward, done, last_obs_info = env.step(action)
            ep_reward += reward
            episode_memory.store(
                observation=last_observation, action=action, reward=reward, value=value,
            )
            ep_rules.append(action[0])
            ep_nodes.append(action[1])
            ep_rewards.append(reward)
            ep_value_estimates.append(value)
            if time_count == self.args.update_gradients_every or done:
                step = self.global_model.optimizer.iterations
                if done and self.args.print_training and self.worker_idx == 0:
                    env.render(last_action=action, last_reward=last_reward)

                self.update_global_network(done, observation, episode_memory)
                self.maybe_write_histograms()
                time_count = 0
                if done:
                    with self.writer.as_default():
                        # Track episode stats
                        pairs = [
                            (np.array(ep_rewards), "rewards"),
                            (np.array(ep_rules), "rules"),
                            (np.array(ep_nodes), "nodes"),
                            (np.array(ep_value_estimates), "v_estimates"),
                        ]
                        for val, name in pairs:
                            tf.summary.scalar(
                                f"episode/{name}/min", data=val.min(), step=step
                            )
                            tf.summary.scalar(
                                f"episode/{name}/max", data=val.max(), step=step
                            )
                            tf.summary.scalar(
                                f"episode/{name}/mean", data=val.mean(), step=step
                            )
                            tf.summary.scalar(
                                f"episode/{name}/std", data=val.std(), step=step
                            )

                    # TODO: histograms of observations and rewards: https://youtu.be/8EcdaCk9KaQ?t=545
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
        window_size: int = self.args.prediction_window_size
        zipped: List[Tuple[MathyWindowObservation, ActionList, RewardList, ValueList]]
        if not done or len(episode_memory.actions) < window_size:
            zipped = episode_memory.to_non_terminal_training_window(window_size)
            # Add fake rewards so the zips look the same
            zipped += (0.0,)
            # Only consider the current window for non-terminal states
            zipped = [zipped]
        else:
            zipped = episode_memory.to_window_observations(
                window=window_size, other_keys=["actions", "rewards", "values"],
            )  # type:ignore

        accumulated_grads: Optional[List[tf.Variable]] = None
        total_losses: AgentLosses = AgentLosses.zero()
        for w, a, r, v in zipped:
            # TODO: can we use the known terminal values in place of rewards
            #       for final episode updates?
            with tf.GradientTape() as tape:
                losses: AgentLosses = self.compute_loss(
                    done=done, inputs=w, actions=a, rewards=r, gamma=self.args.gamma,
                )
                total_losses.accumulate(losses)
            grads = tape.gradient(
                total_losses.total, self.local_model.trainable_weights
            )
            # Accumulate grads for optimizer
            if accumulated_grads is None:
                accumulated_grads = [tf.Variable(f) for f in grads]
            else:
                assert len(grads) == len(accumulated_grads), "gradients must match"
                variable: tf.Variable
                for variable, new_variable in zip(accumulated_grads, grads):
                    variable.assign_add(new_variable)

        with self.writer.as_default():
            step = self.global_model.optimizer.iterations
            prefix = self.tb_prefix
            tf.summary.scalar(
                f"losses/{prefix}/fn_policy_loss",
                data=total_losses.fn_policy,
                step=step,
            )
            tf.summary.scalar(
                f"losses/{prefix}/args_policy_loss",
                data=total_losses.args_policy,
                step=step,
            )
            tf.summary.scalar(
                f"losses/{prefix}/fn_entropy_loss",
                data=total_losses.fn_entropy,
                step=step,
            )
            tf.summary.scalar(
                f"losses/{prefix}/args_entropy_loss",
                data=total_losses.args_entropy,
                step=step,
            )
            tf.summary.scalar(
                f"losses/{prefix}/value_loss", data=total_losses.value, step=step
            )
            tf.summary.scalar(
                f"settings/learning_rate", data=self.optimizer.lr, step=step
            )
            tf.summary.scalar(
                f"{self.tb_prefix}/total_loss", data=total_losses.total, step=step
            )

        # Calculate local gradients
        self.losses.increment("loss", tf.reduce_mean(total_losses.total))
        self.losses.increment("fn_pi", tf.reduce_mean(total_losses.fn_policy))
        self.losses.increment("args_pi", tf.reduce_mean(total_losses.args_policy))
        self.losses.increment("v", tf.reduce_mean(total_losses.value))
        self.losses.increment("fn_h", tf.reduce_mean(total_losses.fn_entropy))
        self.losses.increment("args_h", tf.reduce_mean(total_losses.args_entropy))

        # Push local gradients to global model
        assert accumulated_grads is not None, "no losses computed or gradients found!"
        zipped_gradients = zip(accumulated_grads, self.global_model.trainable_weights)
        self.optimizer.apply_gradients(zipped_gradients)
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())

        if done:
            episode_memory.clear()

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
        inputs: MathyWindowObservation,
        actions: List[ActionType],
        rewards: List[float],
        gamma: float = 0.99,
    ) -> AgentLosses:
        if done:
            bootstrap_value = 0.0  # terminal
        else:
            _, bootstrap_value = predict_action_value(
                self.local_model, inputs.to_inputs()
            )

        losses: AgentLosses = compute_agent_loss(
            model=self.local_model,
            args=self.args,
            inputs=inputs,
            actions=actions,
            rewards=rewards,
            bootstrap_value=bootstrap_value,
            gamma=gamma,
        )

        return losses
