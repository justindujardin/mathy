import math
import os
import threading
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf

from trfl import discrete_policy_entropy_loss, discrete_policy_gradient_loss

from ..core.expressions import MathTypeKeysMax
from ..features import FEATURE_FWD_VECTORS, calculate_grouping_control_signal
from ..r2d2.episode_memory import EpisodeMemory
from ..r2d2.experience import Experience, ExperienceFrame
from ..state import (
    MathyBatchObservation,
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from ..teacher import Student, Teacher, Topic
from ..util import GameRewards
from .actor_critic_model import ActorCriticModel
from .config import A3CArgs
from .util import record, truncate


class A3CWorker(threading.Thread):

    args: A3CArgs

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
        args: A3CArgs,
        action_size: int,
        global_model: ActorCriticModel,
        optimizer,
        greedy_epsilon: float,
        result_queue: Queue,
        experience_queue: Queue,
        experience: Experience,
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
        self.experience_queue = experience_queue
        self.experience = experience
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

        print(
            f"[#{worker_idx}] epsilon: {self.greedy_epsilon} topics: {self.args.topics}"
        )

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

        episode_memory = EpisodeMemory(self.experience_queue)
        while (
            A3CWorker.global_episode < self.args.max_eps
            and A3CWorker.request_quit is False
        ):
            reward = self.run_episode(episode_memory)
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
                            f"{student.topic}/difficulty", data=difficulty, step=step
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
        last_state: MathyObservation = env.reset()
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = -1

        # Set RNN to 0 state for start of episode
        self.local_model.embedding.reset_rnn_state()
        # TODO: Burn in with fake predictions

        while not done and A3CWorker.request_quit is False:
            # store rnn state for replay training
            rnn_state_h = self.local_model.embedding.state_h.numpy()
            rnn_state_c = self.local_model.embedding.state_c.numpy()
            # named tuples are read-only, so add rnn state to a new copy
            last_state = MathyObservation(
                nodes=last_state.nodes,
                mask=last_state.mask,
                type=last_state.type,
                rnn_state=[rnn_state_h, rnn_state_c],
            )

            # Select a random action from the distribution with the given probabilities
            probs, value = self.local_model.predict_next(
                observations_to_window([last_state])
            )

            # store rnn state for replay training
            rnn_state_h = self.local_model.embedding.state_h.numpy()
            rnn_state_c = self.local_model.embedding.state_c.numpy()

            if np.random.random() < self.greedy_epsilon:
                # Select a random action
                action_mask = last_state.mask[:]
                # normalize all valid action to equal probability
                actions = action_mask / np.sum(action_mask)
                action = np.random.choice(len(actions), p=actions)
            else:
                # action = np.random.choice(len(probs), p=probs)
                action = np.argmax(probs)

            # Take an env step
            new_state, reward, done, _ = env.step(action)
            new_state = MathyObservation(
                nodes=new_state.nodes,
                mask=new_state.mask,
                type=new_state.type,
                rnn_state=[rnn_state_h, rnn_state_c],
            )

            new_text = env.state.agent.problem
            grouping_change = calculate_grouping_control_signal(last_text, new_text)
            ep_reward += reward
            frame = ExperienceFrame(
                state=last_state,
                reward=reward,
                action=action,
                terminal=done,
                grouping_change=grouping_change,
                last_action=last_action,
                last_reward=last_reward,
                rnn_state=[rnn_state_h, rnn_state_c],
            )
            episode_memory.store(
                last_state, action, reward, value, grouping_change, frame
            )

            # self.experience.add_frame(frame)
            self.maybe_write_histograms()

            # If the agent has enough experience, train aux tasks
            # if self.experience.is_full() is True:
            #     print("yay")

            if time_count == self.args.update_freq or done:
                self.update_global_network(done, new_state, episode_memory)
                time_count = 0
                if done:
                    self.finish_episode(ep_reward, ep_steps, env.state)

            ep_steps += 1
            time_count += 1
            last_state = new_state
            last_action = action
            last_reward = reward

            # The greedy worker sleeps for a shorter period of time
            sleep = self.args.worker_wait
            if self.worker_idx == 0:
                sleep = sleep // 100
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

            if self.worker_idx == 0:
                # Track global model metrics
                tf.summary.scalar(
                    f"rewards/mean_episode_reward",
                    data=A3CWorker.global_moving_average_reward,
                    step=step,
                )

    def maybe_write_histograms(self):
        if self.worker_idx != 0:
            return
        # The global step is incremented when the optimizer is applied, so check
        # and print summary data here.
        summary_interval = 100
        with self.writer.as_default():
            with tf.summary.record_if(
                lambda: tf.math.equal(
                    self.global_model.global_step % summary_interval, 0
                )
            ):
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.global_step
                    )

    def update_global_network(self, done, new_state, episode_memory: EpisodeMemory):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            loss_tuple = self.compute_loss(
                done, new_state, episode_memory, self.args.gamma
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
        self.optimizer.iterations = self.global_model.global_step
        self.optimizer.apply_gradients(
            zip(grads, self.global_model.trainable_weights),
            global_step=self.global_model.global_step,
        )
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
        new_state: MathyObservation,
        episode_memory: EpisodeMemory,
        gamma=0.99,
    ):
        step = self.global_model.global_step
        if done:
            reward_sum = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values, _ = self.local_model(observations_to_window([new_state]))
            # Select the last timestep
            values = values[-1]
            reward_sum = tf.squeeze(values).numpy()

        # figure out discounted rewards
        discounted_rewards: List[float] = []
        # iterate backwards
        for reward in episode_memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(
            value=np.array(discounted_rewards)[:, None], dtype=tf.float32
        )

        batch_size = len(episode_memory.actions)
        inputs = episode_memory.to_episode_window()
        logits, values, trimmed_logits = self.local_model(inputs, apply_mask=False)
        logits = tf.reshape(logits, [batch_size, -1])
        masked_flat = tf.reshape(trimmed_logits, [batch_size, -1])

        # Calculate entropy and policy loss
        h_loss = discrete_policy_entropy_loss(logits, normalise=True)
        # Scale entropy loss down
        entropy_loss = h_loss.loss * self.args.entropy_loss_scaling
        # pi_loss = discrete_policy_gradient_loss(logits, action_labels, reward_values)

        # Advantage is the difference between the final calculated discount
        # rewards, and the current Value function prediction of the rewards
        advantage = discounted_rewards - values

        # Value loss
        value_loss = advantage ** 2

        # Policy Loss

        # We calculate policy loss from the masked logits to keep
        # the error from exploding when irrelevant (masked) logits
        # have large values. Because we apply a mask for all operations
        # we don't care what those logits are, unless they're part of
        # the mask.
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=episode_memory.actions, logits=masked_flat
        )
        policy_loss *= tf.stop_gradient(advantage)

        value_loss *= 0.5
        total_loss = tf.reduce_mean(value_loss + policy_loss + entropy_loss)
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/policy_loss",
            data=tf.reduce_mean(policy_loss),
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/value_loss",
            data=tf.reduce_mean(value_loss),
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/entropy_loss",
            data=tf.reduce_mean(entropy_loss),
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/advantage",
            data=tf.reduce_sum(advantage),
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/entropy",
            data=tf.reduce_mean(h_loss.extra.entropy),
            step=step,
        )

        return (
            tf.reduce_mean(policy_loss),
            tf.reduce_mean(value_loss),
            tf.reduce_mean(entropy_loss),
            total_loss,
            discounted_rewards,
        )

    def compute_grouping_change_loss(
        self, done, new_state, episode_memory: EpisodeMemory, clip: bool = True
    ):
        change_signals = [signal for signal in episode_memory.grouping_changes]
        signals_tensor = tf.convert_to_tensor(change_signals)
        loss = tf.reduce_mean(signals_tensor)
        if clip is True:
            loss = tf.clip_by_value(loss, -1.0, 1.0)
        return loss

    def rp_samples(self, max_samples=2) -> Tuple[MathyWindowObservation, float]:
        output = MathyWindowObservation([], [], [], [])
        reward: float = 0.0
        if self.experience.is_full() is False:
            return output, reward

        frames = self.experience.sample_rp_sequence()
        states = [frame.state for frame in frames[:-1]]
        target_reward = frames[-1].reward
        if math.isclose(target_reward, GameRewards.TIMESTEP):
            sample_label = 0  # zero
        elif target_reward > 0:
            sample_label = 1  # positive
        else:
            sample_label = 2  # negative
        return observations_to_window(states), sample_label

    def compute_reward_prediction_loss(
        self, done, new_state, episode_memory: EpisodeMemory
    ):
        if not self.experience.is_full():
            return tf.constant(0.0)
        input, label = self.rp_samples()
        rp_output = self.local_model.predict_next_reward(input)
        rp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=rp_output, labels=label
        )
        return tf.reduce_mean(tf.convert_to_tensor(rp_loss))

    def compute_value_replay_loss(self, done, new_state, episode_memory: EpisodeMemory):
        if not self.experience.is_full():
            return tf.constant(0.0)
        sample_size = 6
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
        self, done: bool, new_state, episode_memory: EpisodeMemory, gamma=0.99
    ):
        with self.writer.as_default():
            step = self.global_model.global_step
            loss_tuple = self.compute_policy_value_loss(done, new_state, episode_memory)
            pi_loss, v_loss, h_loss, total_loss, discounted_rewards = loss_tuple
            aux_losses = {}
            episode_memory.commit_frames(discounted_rewards)
            if self.args.use_grouping_control:
                gc_loss = self.compute_grouping_change_loss(
                    done, new_state, episode_memory
                )
                # gc_loss *= aux_weight
                total_loss += gc_loss
                aux_losses["gc"] = gc_loss
            if self.experience.is_full():
                aux_weight = 0.2
                if self.args.use_reward_prediction:
                    rp_loss = self.compute_reward_prediction_loss(
                        done, new_state, episode_memory
                    )
                    rp_loss *= aux_weight
                    total_loss += rp_loss
                    aux_losses["rp"] = rp_loss
                if self.args.use_value_replay:
                    vr_loss = self.compute_value_replay_loss(
                        done, new_state, episode_memory
                    )
                    vr_loss *= aux_weight
                    total_loss += vr_loss
                    aux_losses["vr"] = vr_loss
                for key in aux_losses.keys():
                    tf.summary.scalar(
                        f"worker_{self.worker_idx}/losses/{key}_loss",
                        data=aux_losses[key],
                        step=step,
                    )

            tf.summary.scalar(
                f"worker_{self.worker_idx}/losses/total_loss",
                data=total_loss,
                step=step,
            )

        return pi_loss, v_loss, h_loss, aux_losses, total_loss

    # Auxiliary tasks
    def get_rp_inputs_with_labels(
        self, episode_buffer: EpisodeMemory
    ) -> Tuple[List[Any], List[Any]]:
        # [Reward prediction]
        rp_experience_frames: List[
            ExperienceFrame
        ] = self.experience.sample_rp_sequence()
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
