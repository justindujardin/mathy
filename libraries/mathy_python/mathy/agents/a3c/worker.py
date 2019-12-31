import math
import os
import queue
import threading
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np
import tensorflow as tf
from wasabi import msg

from ...util import print_error

from ...envs.gym.mathy_gym_env import MathyGymEnv
from ...state import (
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    ObservationFeatureIndices,
    observations_to_window,
)
from ...teacher import Teacher
from ...util import calculate_grouping_control_signal, discount
from .. import action_selectors
from ..episode_memory import EpisodeMemory
from ..mcts import MCTS
from ..policy_value_model import PolicyValueModel, get_or_create_policy_model
from ..trfl import discrete_policy_entropy_loss, td_lambda
from .config import A3CConfig
from .util import record, truncate


class A3CWorker(threading.Thread):

    args: A3CConfig

    # <GLOBAL_VARS>
    global_episode = 0
    global_moving_average_reward = 0
    save_every_n_episodes = 50
    request_quit = False
    save_lock = threading.Lock()
    # </GLOBAL_VARS>

    envs: Dict[str, Any]

    def __init__(
        self,
        args: A3CConfig,
        action_size: int,
        global_model: PolicyValueModel,
        optimizer,
        greedy_epsilon: Union[float, List[float]],
        result_queue: Queue,
        worker_idx: int,
        writer: tf.summary.SummaryWriter,
        teacher: Teacher,
    ):
        super(A3CWorker, self).__init__()
        self.args = args
        self.greedy_epsilon = greedy_epsilon
        self.iteration = 0
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.optimizer = optimizer
        self.worker_idx = worker_idx
        self.teacher = teacher
        self.envs = {}

        with msg.loading(f"Worker {worker_idx} starting..."):
            first_env = self.teacher.get_env(self.worker_idx, self.iteration)
            self.envs[first_env] = gym.make(first_env)
            self.writer = writer
            self.local_model = get_or_create_policy_model(args, self.action_size)
            self.reset_episode_loss()
            self.last_model_write = -1
            self.last_histogram_write = -1
        msg.good(f"Worker {worker_idx} started.")

    @property
    def tb_prefix(self) -> str:
        if self.worker_idx == 0:
            return "agent"
        return f"workers/{self.worker_idx}"

    @property
    def epsilon(self) -> float:
        """Return an exploration epsilon for use in an episode"""
        e = 0.0
        if self.worker_idx == 0 and self.args.main_worker_use_epsilon is False:
            return e

        if isinstance(self.greedy_epsilon, list):
            e = np.random.choice(self.greedy_epsilon)
        elif isinstance(self.greedy_epsilon, float):
            e = self.greedy_epsilon
        else:
            raise ValueError("greedy_epsilon must either be a float or list of floats")
        e = truncate(e)
        return e

    def reset_episode_loss(self):
        self.ep_loss = 0.0
        self.ep_pi_loss = 0.0
        self.ep_value_loss = 0.0
        self.ep_aux_loss: Dict[str, float] = {}
        self.ep_entropy_loss = 0.0

    def run(self):
        if self.args.profile:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        episode_memory = EpisodeMemory()
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
        self.result_queue.put(None)

    def build_episode_selector(
        self, env: MathyGymEnv
    ) -> "action_selectors.ActionSelector":
        if self.worker_idx == 0:
            # disable dirichlet noise in worker_0
            epsilon = 0.0
        else:
            # explore based on eGreedy param (wild guess for values)
            epsilon = 0.1 + self.epsilon
        selector: action_selectors.ActionSelector
        if self.args.action_strategy == "mcts_worker_0":
            mcts = MCTS(
                env=env.mathy,
                model=self.local_model,
                num_mcts_sims=self.args.mcts_sims,
                epsilon=epsilon,
            )
            if self.worker_idx == 0:
                selector = action_selectors.MCTSActionSelector(
                    model=self.global_model,
                    worker_id=self.worker_idx,
                    mcts=mcts,
                    episode=A3CWorker.global_episode,
                )
            else:
                selector = action_selectors.A3CEpsilonGreedyActionSelector(
                    model=self.global_model,
                    worker_id=self.worker_idx,
                    epsilon=self.epsilon,
                    episode=A3CWorker.global_episode,
                )
        elif self.args.action_strategy == "mcts_worker_n":
            mcts = MCTS(
                env=env.mathy,
                model=self.local_model,
                num_mcts_sims=self.args.mcts_sims,
                epsilon=epsilon,
            )
            if self.worker_idx != 0:
                selector = action_selectors.MCTSActionSelector(
                    model=self.global_model,
                    worker_id=self.worker_idx,
                    mcts=mcts,
                    episode=A3CWorker.global_episode,
                )
            else:
                selector = action_selectors.A3CEpsilonGreedyActionSelector(
                    model=self.global_model,
                    worker_id=self.worker_idx,
                    epsilon=self.epsilon,
                    episode=A3CWorker.global_episode,
                )
        elif self.args.action_strategy in ["a3c"]:
            selector = action_selectors.A3CEpsilonGreedyActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
                epsilon=self.epsilon,
                episode=A3CWorker.global_episode,
            )
        else:
            raise EnvironmentError(
                f"Unknown action_strategy: {self.args.action_strategy}"
            )
        return selector

    def run_episode(self, episode_memory: EpisodeMemory):
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)
        if env_name not in self.envs:
            self.envs[env_name] = gym.make(env_name)
        env = self.envs[env_name]
        episode_memory.clear()
        self.ep_loss = 0
        ep_reward = 0.0
        ep_steps = 1
        time_count = 0
        done = False
        last_observation: MathyObservation = env.reset(rnn_size=self.args.lstm_units)
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = -1

        selector = self.build_episode_selector(env)

        # Set RNN to 0 state for start of episode
        selector.model.embedding.reset_rnn_state()

        # Start with the "init" sequence [n] times
        for i in range(self.args.num_thinking_steps_begin):
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())
            seq_start = env.state.to_start_observation(rnn_state_h, rnn_state_c)
            try:
                window = observations_to_window([seq_start, last_observation])
                selector.model.call(window.to_inputs())
            except BaseException as err:
                print_error(
                    err, f"Episode begin thinking steps prediction failed.",
                )
                continue

        while not done and A3CWorker.request_quit is False:
            if self.args.print_training and self.worker_idx == 0:
                env.render(self.args.print_mode, None)
            # store rnn state for replay training
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())
            last_rnn_state = [rnn_state_h, rnn_state_c]

            # named tuples are read-only, so add rnn state to a new copy
            last_observation = MathyObservation(
                nodes=last_observation.nodes,
                mask=last_observation.mask,
                values=last_observation.values,
                type=last_observation.type,
                time=last_observation.time,
                rnn_state_h=tf.squeeze(rnn_state_h),
                rnn_state_c=tf.squeeze(rnn_state_c),
                rnn_history_h=episode_memory.rnn_weighted_history(self.args.lstm_units)[
                    0
                ],
            )
            # before_rnn_state_h = selector.model.embedding.state_h.numpy()
            # before_rnn_state_c = selector.model.embedding.state_c.numpy()

            window = episode_memory.to_window_observation(last_observation)
            try:
                action, value = selector.select(
                    last_state=env.state,
                    last_window=window,
                    last_action=last_action,
                    last_reward=last_reward,
                    last_rnn_state=last_rnn_state,
                )
            except BaseException as err:
                print_error(err, "failed to select an action during an episode step")
                continue

            # Take an env step
            observation, reward, done, _ = env.step(action)
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())

            # TODO: make this a unit test, check that EpisodeMemory states are not equal
            #       across time steps.
            # compare_states_h = tf.math.equal(before_rnn_state_h,rnn_state_h)
            # compare_states_c = tf.math.equal(before_rnn_state_h,rnn_state_h)
            # assert before_rnn_state_h != rnn_state_h
            # assert before_rnn_state_c != rnn_state_c

            observation = MathyObservation(
                nodes=observation.nodes,
                mask=observation.mask,
                values=observation.values,
                type=observation.type,
                time=observation.time,
                rnn_state_h=rnn_state_h,
                rnn_state_c=rnn_state_c,
                rnn_history_h=episode_memory.rnn_weighted_history(self.args.lstm_units)[
                    0
                ],
            )

            new_text = env.state.agent.problem
            grouping_change = calculate_grouping_control_signal(
                last_text, new_text, clip_at_zero=self.args.clip_grouping_control
            )
            ep_reward += reward
            episode_memory.store(
                observation=last_observation,
                action=action,
                reward=reward,
                grouping_change=grouping_change,
                value=value,
            )
            if time_count == self.args.update_gradients_every or done:
                if done and self.args.print_training and self.worker_idx == 0:
                    env.render(self.args.print_mode, None)

                # TODO: Make this a unit test?
                # Check that the LSTM h/c states changed over time in the episode.
                #
                # NOTE: in practice it seems every once in a while the state doesn't
                # change, and I suppose this makes sense if the LSTM thought the
                # existing state was... fine?
                #
                # check_rnn = None
                # for obs in episode_memory.observations:
                #     if check_rnn is not None:
                #         h_equal_indices = (
                #             tf.squeeze(tf.math.equal(obs.rnn_state_h, check_rnn[0]))
                #             .numpy()
                #             .tolist()
                #         )
                #         c_equal_indices = (
                #             tf.squeeze(tf.math.equal(obs.rnn_state_c, check_rnn[1]))
                #             .numpy()
                #             .tolist()
                #         )
                #         assert False in h_equal_indices
                #         assert False in c_equal_indices

                #     check_rnn = [obs.rnn_state_h, obs.rnn_state_c]

                self.update_global_network(done, observation, episode_memory)
                self.maybe_write_histograms()
                time_count = 0
                if done:
                    self.finish_episode(ep_reward, ep_steps, env.state)

            ep_steps += 1
            time_count += 1
            last_observation = observation
            last_action = int(action)
            last_reward = reward

            # If there are multiple workers, apply a worker sleep
            # to give the system some breathing room.
            if self.args.num_workers > 1:
                # The greedy worker sleeps for a shorter period of time
                sleep = self.args.worker_wait
                if self.worker_idx == 0:
                    sleep = max(sleep // 100, 0.05)
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

    def maybe_write_histograms(self):
        if self.worker_idx != 0:
            return
        step = self.global_model.optimizer.iterations.numpy()
        next_write = self.last_histogram_write + self.args.summary_interval
        if step >= next_write or self.last_histogram_write == -1:
            with self.writer.as_default():
                self.last_histogram_write = step
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.optimizer.iterations
                    )
                # Write out current LSTM hidden/cell states
                tf.summary.histogram(
                    "memory/lstm_c",
                    self.local_model.embedding.state_c,
                    step=self.global_model.optimizer.iterations,
                )
                tf.summary.histogram(
                    "memory/lstm_h",
                    self.local_model.embedding.state_h,
                    step=self.global_model.optimizer.iterations,
                )

    def update_global_network(
        self, done: bool, observation: MathyObservation, episode_memory: EpisodeMemory,
    ):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            loss_tuple = self.compute_loss(
                done=done,
                observation=observation,
                episode_memory=episode_memory,
                gamma=self.args.gamma,
            )
            pi_loss, value_loss, entropy_loss, aux_losses, total_loss = loss_tuple
        self.ep_loss += total_loss
        self.ep_pi_loss += pi_loss
        self.ep_value_loss += value_loss
        self.ep_entropy_loss += entropy_loss
        for k in aux_losses.keys():
            if k not in self.ep_aux_loss:
                self.ep_aux_loss[k] = 0.0
            self.ep_aux_loss[k] += aux_losses[k].numpy()
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # Push local gradients to global model

        zipped_gradients = zip(grads, self.global_model.trainable_weights)
        # Assert that we always have some gradient flow in each trainable var

        # TODO: Make this a unit test. It degrades performance at train time
        # for grad, var in zipped_gradients:
        #     nonzero_grads = tf.math.count_nonzero(grad).numpy()
        #     grad_sum = tf.math.reduce_sum(grad).numpy()
        #     # if "lstm" in var.name and self.worker_idx == 0:
        #     #     print(f"[{var.name}] {grad_sum}")
        #     if nonzero_grads == 0:
        #         tf.print(grad_sum)
        #         raise ValueError(f"{var.name} has no gradient")

        self.optimizer.apply_gradients(zipped_gradients)
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())
        episode_memory.clear()

    def finish_episode(self, episode_reward, episode_steps, last_state: MathyEnvState):
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)

        # Only observe/track the most-greedy worker (high epsilon exploration
        # stats are unlikely to be consistent or informative)
        if self.worker_idx == 0:
            A3CWorker.global_moving_average_reward = record(
                A3CWorker.global_episode,
                episode_reward,
                self.worker_idx,
                A3CWorker.global_moving_average_reward,
                self.result_queue,
                self.ep_pi_loss,
                self.ep_value_loss,
                self.ep_entropy_loss,
                self.ep_aux_loss,
                self.ep_loss,
                episode_steps,
                env_name,
            )
            self.maybe_write_episode_summaries(
                episode_reward, episode_steps, last_state
            )

            step = self.global_model.optimizer.iterations.numpy()
            next_write = self.last_model_write + A3CWorker.save_every_n_episodes
            if step >= next_write or self.last_model_write == -1:
                self.last_model_write = step
                self.write_global_model()

        A3CWorker.global_episode += 1
        self.reset_episode_loss()

    def write_global_model(self, increment_episode=True):
        with A3CWorker.save_lock:
            # Do this inside the lock so other threads can't also acquire the
            # lock in the time between when it's released and assigned outside
            # of the if conditional.
            if increment_episode is True:
                A3CWorker.global_episode += 1
                self.global_model.save()

    def compute_policy_value_loss(
        self,
        done: bool,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        gamma=0.99,
    ):
        step = self.global_model.optimizer.iterations
        if done:
            bootstrap_value = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values, _ = self.local_model.call(
                observations_to_window([observation]).to_inputs()
            )
            # Select the last timestep
            values = values[-1]
            bootstrap_value = tf.squeeze(values).numpy()

        discounted_rewards: List[float] = []
        for reward in episode_memory.rewards[::-1]:
            bootstrap_value = reward + gamma * bootstrap_value
            discounted_rewards.append(bootstrap_value)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(
            value=np.array(discounted_rewards)[:, None], dtype=tf.float32
        )

        batch_size = len(episode_memory.actions)
        sequence_length = len(episode_memory.observations[0].nodes)
        inputs = episode_memory.to_episode_window().to_inputs()
        logits, values, trimmed_logits = self.local_model(inputs, apply_mask=False)

        logits = tf.reshape(logits, [batch_size, -1])
        policy_logits = tf.reshape(trimmed_logits, [batch_size, -1])

        # Calculate entropy and policy loss
        h_loss = discrete_policy_entropy_loss(
            logits, normalise=self.args.normalize_entropy_loss
        )
        # Scale entropy loss down
        entropy_loss = h_loss.loss * self.args.entropy_loss_scaling
        entropy_loss = tf.reduce_mean(entropy_loss)

        rewards_tensor = tf.convert_to_tensor(episode_memory.rewards, dtype=tf.float32)
        rewards_tensor = tf.expand_dims(rewards_tensor, 1)
        pcontinues = tf.convert_to_tensor([[gamma]] * batch_size, dtype=tf.float32)
        bootstrap_value = tf.convert_to_tensor([bootstrap_value], dtype=tf.float32)

        lambda_loss = td_lambda(
            state_values=values,
            rewards=rewards_tensor,
            pcontinues=pcontinues,
            bootstrap_value=bootstrap_value,
            lambda_=self.args.td_lambda,
        )
        advantage = lambda_loss.extra.temporal_differences
        # Value loss
        value_loss = tf.reduce_mean(lambda_loss.loss)

        # Policy Loss
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=episode_memory.actions, logits=policy_logits
        )

        policy_loss *= advantage
        policy_loss = tf.reduce_mean(policy_loss)

        # Scale the policy/value losses down by the sequence length to normalize
        # for combination with aux losses.
        policy_loss /= sequence_length
        # value_loss /= sequence_length

        total_loss = value_loss + policy_loss + entropy_loss
        prefix = self.tb_prefix
        tf.summary.scalar(f"{prefix}/policy_loss", data=policy_loss, step=step)
        tf.summary.scalar(f"{prefix}/value_loss", data=value_loss, step=step)
        tf.summary.scalar(f"{prefix}/entropy_loss", data=entropy_loss, step=step)
        tf.summary.scalar(
            f"{prefix}/advantage", data=tf.reduce_mean(advantage), step=step
        )
        tf.summary.scalar(
            f"{prefix}/entropy", data=tf.reduce_mean(h_loss.extra.entropy), step=step
        )

        return (policy_loss, value_loss, entropy_loss, total_loss, discounted_rewards)

    def compute_grouping_change_loss(
        self,
        done,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        clip: bool = True,
    ):
        change_signals = [signal for signal in episode_memory.grouping_changes]
        signals_tensor = tf.convert_to_tensor(change_signals)
        loss = tf.reduce_mean(signals_tensor)
        if clip is True:
            loss = tf.clip_by_value(loss, -1.0, 1.0)
        return loss

    def compute_loss(
        self,
        *,
        done: bool,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        gamma=0.99,
    ):
        with self.writer.as_default():
            step = self.global_model.optimizer.iterations
            loss_tuple = self.compute_policy_value_loss(
                done, observation, episode_memory
            )
            pi_loss, v_loss, h_loss, total_loss, discounted_rewards = loss_tuple
            aux_losses = {}
            aux_weight = self.args.aux_tasks_weight_scale

            if self.args.use_grouping_control:
                gc_loss = self.compute_grouping_change_loss(
                    done,
                    observation,
                    episode_memory,
                    clip=self.args.clip_grouping_control,
                )
                gc_loss *= aux_weight
                total_loss += gc_loss
                aux_losses["gc"] = gc_loss
            for key in aux_losses.keys():
                tf.summary.scalar(
                    f"{self.tb_prefix}/{key}_loss", data=aux_losses[key], step=step
                )

            tf.summary.scalar(
                f"{self.tb_prefix}/total_loss", data=total_loss, step=step
            )

        return pi_loss, v_loss, h_loss, aux_losses, total_loss
