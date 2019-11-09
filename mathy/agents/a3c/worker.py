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

from ...features import calculate_grouping_control_signal
from ...gym.mathy_gym_env import MathyGymEnv
from ...state import (
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from ...teacher import Teacher
from ...util import GameRewards, discount
from .. import action_selectors
from ..actor_critic_model import ActorCriticModel
from ..base_config import BaseConfig
from ..episode_memory import EpisodeMemory
from ..experience import Experience, ExperienceFrame
from ..mcts import MCTS
from ..tensorflow.trfl import discrete_policy_entropy_loss, td_lambda
from .util import record, truncate


class A3CWorker(threading.Thread):

    args: BaseConfig

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
        args: BaseConfig,
        action_size: int,
        global_model: ActorCriticModel,
        optimizer,
        greedy_epsilon: Union[float, List[float]],
        result_queue: Queue,
        experience_queue: Queue,
        cmd_queue: Queue,
        worker_idx: int,
        writer: tf.summary.SummaryWriter,
        teacher: Teacher,
        is_actor: bool = True,
        is_learner: bool = True,
    ):
        super(A3CWorker, self).__init__()
        self.args = args
        self.greedy_epsilon = greedy_epsilon
        self.iteration = 0
        self.action_size = action_size
        self.result_queue = result_queue
        self.experience_queue = experience_queue
        self.cmd_queue = cmd_queue
        history_size = self.args.history_size
        if worker_idx == 0:
            history_size = self.args.greedy_history_size
        self.experience = Experience(
            history_size=history_size, ready_at=self.args.ready_at
        )
        self.global_model = global_model
        self.optimizer = optimizer
        self.worker_idx = worker_idx
        self.teacher = teacher
        self.envs = {}
        first_env = self.teacher.get_env(self.worker_idx, self.iteration)
        self.envs[first_env] = gym.make(first_env)
        self.writer = writer
        self.local_model = ActorCriticModel(
            args=args, predictions=self.action_size, optimizer=self.optimizer
        )
        self.local_model.maybe_load(
            self.envs[first_env].initial_window(self.args.lstm_units)
        )
        self.reset_episode_loss()
        self.last_histogram_write = -1

        # Sanity check
        if self.args.action_strategy == "mcts_e_unreal":
            if self.args.num_workers == 1:
                raise EnvironmentError(
                    "You are attempting to use MCTS with an UNREAL style agent, "
                    "but you have only enabled 1 worker. This configuration requires "
                    "at minimum 2 workers, because MCTS is not run on worker_0. This "
                    "is to keep the true Greedy worker training/reporting outputs "
                    "consistently. It's also helpful to see the strength of the model "
                    "without the tree search. We don't want to ship a 1000x slower "
                    "model at runtime just to solve basic problems."
                )

        print(f"[#{worker_idx}] e: {self.epsilon} topics: {self.args.topics}")

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

        episode_memory = EpisodeMemory(self.experience, self.experience_queue)
        while (
            A3CWorker.global_episode < self.args.max_eps
            and A3CWorker.request_quit is False
        ):
            try:
                ctrl, data = self.cmd_queue.get_nowait()
                if ctrl == "experience":
                    for frame in data:
                        self.experience.add_frame(frame)
                    # total = self.experience.frame_count
                    # print(f"[#{self.worker_idx}] in({len(data)}) total({total})")
            except queue.Empty:
                pass

            reward = self.run_episode(episode_memory)
            if (
                A3CWorker.global_episode
                > self.args.teacher_start_evaluations_at_episode
            ):
                win_pct = self.teacher.report_result(self.worker_idx, reward)
                if win_pct is not None:
                    with self.writer.as_default():
                        student = self.teacher.get_student(self.worker_idx)
                        difficulty = student.topics[student.topic].difficulty
                        if difficulty == "easy":
                            difficulty = 0.0
                        elif difficulty == "normal":
                            difficulty = 0.5
                        elif difficulty == "hard":
                            difficulty = 1.0
                        step = self.global_model.global_step
                        if self.worker_idx == 0:
                            tf.summary.scalar(
                                f"{student.topic}/success_rate", data=win_pct, step=step
                            )
                            tf.summary.scalar(
                                f"{student.topic}/difficulty",
                                data=difficulty,
                                step=step,
                            )

            self.iteration += 1
            # TODO: Make this a subprocess? Python threads won't scale up well to
            #       many cores, I think.

        if self.args.profile:
            profile_name = f"worker_{self.worker_idx}.profile"
            profile_path = os.path.join(self.args.model_dir, profile_name)
            pr.disable()
            pr.dump_stats(profile_path)
            print(f"PROFILER: saved {profile_path}")
        self.result_queue.put(None)

    def build_episode_selector(
        self, env: MathyGymEnv
    ) -> "action_selectors.ActionSelector":
        mcts: Optional[MCTS] = None
        if "mcts" in self.args.action_strategy:
            if self.worker_idx == 0:
                # disable dirichlet noise in worker_0
                epsilon = 0.0
            else:
                # explore based on eGreedy param (wild guess for values)
                epsilon = 0.1 + self.epsilon
            mcts = MCTS(
                env=env.mathy,
                model=self.local_model,
                num_mcts_sims=self.args.mcts_sims,
                epsilon=epsilon,
            )
        selector: action_selectors.ActionSelector
        if mcts is not None and self.args.action_strategy == "mcts_e_unreal":
            if (
                self.args.use_reward_prediction is False
                and self.args.use_value_replay is False
            ):
                raise EnvironmentError(
                    "This model is not configured for reward prediction/value replay. "
                    "To use UNREAL MCTS you must enable at least one UNREAL aux task."
                )

            selector = action_selectors.UnrealMCTSActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
                mcts=mcts,
                epsilon=self.args.unreal_mcts_epsilon,
                episode=A3CWorker.global_episode,
            )
        elif mcts is not None and self.args.action_strategy == "mcts":
            selector = action_selectors.MCTSActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
                mcts=mcts,
                episode=A3CWorker.global_episode,
            )
        elif mcts is not None and self.args.action_strategy == "mcts_worker_0":
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
        elif mcts is not None and self.args.action_strategy == "mcts_worker_n":
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
        elif mcts is not None and self.args.action_strategy == "mcts_recover":
            selector = action_selectors.MCTSRecoveryActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
                mcts=mcts,
                episode=A3CWorker.global_episode,
                recover_threshold=self.args.mcts_recover_time_threshold,
                base_selector=action_selectors.A3CEpsilonGreedyActionSelector(
                    model=self.global_model,
                    worker_id=self.worker_idx,
                    epsilon=self.epsilon,
                    episode=A3CWorker.global_episode,
                ),
            )
        elif self.args.action_strategy in ["a3c", "unreal"]:
            selector = action_selectors.A3CEpsilonGreedyActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
                epsilon=self.epsilon,
                episode=A3CWorker.global_episode,
            )
        elif self.args.action_strategy == "a3c-eval":
            selector = action_selectors.A3CGreedyActionSelector(
                model=self.global_model,
                worker_id=self.worker_idx,
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
            rnn_state_h = selector.model.embedding.state_h.numpy()
            rnn_state_c = selector.model.embedding.state_c.numpy()
            seq_start = env.start_observation([rnn_state_h, rnn_state_c])
            selector.model.predict_next(
                observations_to_window([seq_start, last_observation])
            )

        while not done and A3CWorker.request_quit is False:
            if self.args.print_training and self.worker_idx == 0:
                env.render(
                    self.args.print_mode, selector.model.embedding.attention_weights
                )
            # store rnn state for replay training
            rnn_state_h = selector.model.embedding.state_h.numpy()
            rnn_state_c = selector.model.embedding.state_c.numpy()
            last_rnn_state = [rnn_state_h, rnn_state_c]

            # named tuples are read-only, so add rnn state to a new copy
            last_observation = MathyObservation(
                nodes=last_observation.nodes,
                mask=last_observation.mask,
                hints=last_observation.hints,
                type=last_observation.type,
                time=last_observation.time,
                rnn_state=last_rnn_state,
                rnn_history=episode_memory.rnn_weighted_history(self.args.lstm_units),
            )
            # before_rnn_state_h = selector.model.embedding.state_h.numpy()
            # before_rnn_state_c = selector.model.embedding.state_c.numpy()

            window = episode_memory.to_window_observation(last_observation)
            action, value = selector.select(
                last_state=env.state,
                last_window=window,
                last_action=last_action,
                last_reward=last_reward,
                last_rnn_state=last_rnn_state,
            )
            # Take an env step
            observation, reward, done, _ = env.step(action)
            rnn_state_h = selector.model.embedding.state_h.numpy()
            rnn_state_c = selector.model.embedding.state_c.numpy()

            # TODO: make this a unit test, check that EpisodeMemory states are not equal
            #       across time steps.
            # compare_states_h = tf.math.equal(before_rnn_state_h,rnn_state_h)
            # compare_states_c = tf.math.equal(before_rnn_state_h,rnn_state_h)
            # assert before_rnn_state_h != rnn_state_h
            # assert before_rnn_state_c != rnn_state_c

            observation = MathyObservation(
                nodes=observation.nodes,
                mask=observation.mask,
                hints=observation.hints,
                type=observation.type,
                time=observation.time,
                rnn_state=[rnn_state_h, rnn_state_c],
                rnn_history=episode_memory.rnn_weighted_history(self.args.lstm_units),
            )

            new_text = env.state.agent.problem
            grouping_change = calculate_grouping_control_signal(
                last_text, new_text, clip_at_zero=True
            )
            ep_reward += reward
            frame = ExperienceFrame(
                state=last_observation,
                reward=reward,
                action=action,
                terminal=done,
                grouping_change=grouping_change,
                last_action=last_action,
                last_reward=last_reward,
                rnn_state=[rnn_state_h, rnn_state_c],
            )
            episode_memory.store(
                observation=last_observation,
                action=action,
                reward=reward,
                grouping_change=grouping_change,
                frame=frame,
                value=value,
            )
            if time_count == self.args.update_freq or done:
                keep_experience = bool(
                    self.args.use_value_replay or self.args.use_reward_prediction
                )
                if self.args.action_strategy == "unreal":
                    keep_experience = True
                elif self.args.action_strategy == "mcts_e_unreal":
                    unreal = cast(action_selectors.UnrealMCTSActionSelector, selector)
                    keep_experience = unreal.use_mcts or not self.experience.is_full()

                if done and self.args.print_training and self.worker_idx == 0:
                    env.render(
                        self.args.print_mode,
                        selector.model.embedding.attention_weights,
                    )

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
                #             tf.squeeze(tf.math.equal(obs.rnn_state[0], check_rnn[0]))
                #             .numpy()
                #             .tolist()
                #         )
                #         c_equal_indices = (
                #             tf.squeeze(tf.math.equal(obs.rnn_state[1], check_rnn[1]))
                #             .numpy()
                #             .tolist()
                #         )
                #         assert False in h_equal_indices
                #         assert False in c_equal_indices

                #     check_rnn = obs.rnn_state

                self.update_global_network(
                    done, observation, episode_memory, keep_experience=keep_experience
                )
                self.maybe_write_histograms()
                time_count = 0
                if done:
                    self.finish_episode(ep_reward, ep_steps, env.state)

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
                    sleep = max(sleep // 100, 0.05)
                # Workers wait between each step so that it's possible
                # to run more workers than there are CPUs available.
                time.sleep(sleep)
        return ep_reward

    def maybe_write_episode_summaries(
        self, episode_reward: float, episode_steps: int, last_state: MathyEnvState
    ):
        if self.worker_idx != 0:
            return

        # Track metrics for all workers
        name = self.teacher.get_env(self.worker_idx, self.iteration)
        step = self.global_model.global_step
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
        step = self.global_model.global_step.numpy()
        next_write = self.last_histogram_write + self.args.summary_interval
        if step >= next_write or self.last_histogram_write == -1:
            with self.writer.as_default():
                self.last_histogram_write = step
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.global_step
                    )
                # Write out current LSTM hidden/cell states
                tf.summary.histogram(
                    "agent/lstm_c",
                    self.local_model.embedding.state_c,
                    step=self.global_model.global_step,
                )
                tf.summary.histogram(
                    "agent/lstm_h",
                    self.local_model.embedding.state_h,
                    step=self.global_model.global_step,
                )

    def update_global_network(
        self,
        done: bool,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        keep_experience: bool,
    ):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            loss_tuple = self.compute_loss(
                done=done,
                observation=observation,
                episode_memory=episode_memory,
                gamma=self.args.gamma,
                keep_experience=keep_experience,
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

        # We must use a lock to save our model and to print to prevent data races.
        if A3CWorker.global_episode % A3CWorker.save_every_n_episodes == 0:
            self.write_global_model()
        else:
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
        step = self.global_model.global_step
        if done:
            bootstrap_value = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values, _ = self.local_model(observations_to_window([observation]))
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
        inputs = episode_memory.to_episode_window()
        logits, values, trimmed_logits = self.local_model(inputs, apply_mask=False)

        logits = tf.reshape(logits, [batch_size, -1])
        policy_logits = tf.reshape(trimmed_logits, [batch_size, -1])

        # Calculate entropy and policy loss
        h_loss = discrete_policy_entropy_loss(logits, normalise=True)
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

    def rp_samples(self) -> Tuple[MathyWindowObservation, List[float]]:
        output = MathyWindowObservation(
            nodes=[],
            mask=[],
            hints=[],
            type=[],
            rnn_state=[[], []],
            rnn_history=[[], []],
            time=[],
        )
        reward: float = 0.0
        if self.experience.is_full() is False:
            return output, [reward]

        frames = self.experience.sample_rp_sequence()
        states = [frame.state for frame in frames[:-1]]
        target_reward = frames[-1].reward
        if math.isclose(target_reward, GameRewards.TIMESTEP):
            sample_label = 0  # zero
        elif target_reward > 0:
            sample_label = 1  # positive
        else:
            sample_label = 2  # negative
        return observations_to_window(states), [sample_label]

    def compute_reward_prediction_loss(
        self, done, observation: MathyObservation, episode_memory: EpisodeMemory
    ):
        if not self.experience.is_full():
            return tf.constant(0.0)
        max_steps = 3
        rp_losses = []
        for i in range(max_steps):
            input, label = self.rp_samples()
            rp_output = self.local_model.predict_next_reward(input)
            rp_losses.append(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=rp_output, labels=label
                )
            )
        return tf.reduce_mean(tf.convert_to_tensor(rp_losses))

    def compute_value_replay_loss(
        self, done, observation: MathyObservation, episode_memory: EpisodeMemory
    ):
        if not self.experience.is_full():
            return tf.constant(0.0)
        sample_size = 12
        frames: List[ExperienceFrame] = self.experience.sample_sequence(sample_size)
        states = []
        discounted_rewards = []
        for frame in frames:
            states.append(frame.state)
            discounted_rewards.append(frame.discounted)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)
        observation_window = observations_to_window(states)
        vr_values = self.local_model.predict_value_replays(observation_window)
        advantage = discounted_rewards - vr_values
        # Value loss
        value_loss = advantage ** 2
        return tf.reduce_mean(tf.convert_to_tensor(value_loss))

    def compute_loss(
        self,
        *,
        done: bool,
        observation: MathyObservation,
        episode_memory: EpisodeMemory,
        keep_experience=False,
        gamma=0.99,
    ):
        with self.writer.as_default():
            step = self.global_model.global_step
            loss_tuple = self.compute_policy_value_loss(
                done, observation, episode_memory
            )
            pi_loss, v_loss, h_loss, total_loss, discounted_rewards = loss_tuple
            aux_losses = {}
            use_replay = self.args.use_reward_prediction or self.args.use_value_replay
            # Skip over the experience replay buffers if not using Aux tasks because
            # they add extra memory overhead to hold on to the frames.
            if use_replay is True and keep_experience is True:
                episode_memory.commit_frames(self.worker_idx, discounted_rewards)
            aux_weight = self.args.aux_tasks_weight_scale

            if self.args.use_grouping_control:
                gc_loss = self.compute_grouping_change_loss(
                    done, observation, episode_memory
                )
                gc_loss *= aux_weight
                total_loss += gc_loss
                aux_losses["gc"] = gc_loss
            if self.experience.is_full():
                if self.args.use_reward_prediction:
                    rp_loss = self.compute_reward_prediction_loss(
                        done, observation, episode_memory
                    )
                    rp_loss *= aux_weight
                    total_loss += rp_loss
                    aux_losses["rp"] = rp_loss
                if self.args.use_value_replay:
                    vr_loss = self.compute_value_replay_loss(
                        done, observation, episode_memory
                    )
                    vr_loss *= aux_weight
                    total_loss += vr_loss
                    aux_losses["vr"] = vr_loss
            for key in aux_losses.keys():
                tf.summary.scalar(
                    f"{self.tb_prefix}/{key}_loss", data=aux_losses[key], step=step
                )

            tf.summary.scalar(
                f"{self.tb_prefix}/total_loss", data=total_loss, step=step
            )

        return pi_loss, v_loss, h_loss, aux_losses, total_loss
