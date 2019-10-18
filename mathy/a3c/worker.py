import math
import os
import queue
import threading
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple, cast

import gym
import numpy as np
import tensorflow as tf

from trfl import discrete_policy_entropy_loss, discrete_policy_gradient_loss

from ..agent.mcts import MCTS
from ..core.expressions import MathTypeKeysMax
from ..features import FEATURE_FWD_VECTORS, calculate_grouping_control_signal
from ..gym.mathy_gym_env import MathyGymEnv
from ..r2d2.episode_memory import EpisodeMemory
from ..r2d2.experience import Experience, ExperienceFrame
from ..state import (
    MathyBatchObservation,
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
    windows_to_batch,
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
        cmd_queue: Queue,
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

        e = truncate(self.greedy_epsilon)
        if self.worker_idx == 0 and self.args.main_worker_use_epsilon is False:
            e = 0.0

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

        print(
            f"[#{worker_idx}] e: {e} history: {history_size} topics: {self.args.topics}"
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

    def build_episode_selector(self, env: MathyGymEnv) -> "ActionSelector":
        mcts: Optional[MCTS] = None
        if "mcts" in self.args.action_strategy:
            if self.worker_idx == 0:
                # disable dirichlet noise in worker_0
                epsilon = 0.0
            else:
                # explore based on eGreedy param (wild guess for values)
                epsilon = 0.1 + self.greedy_epsilon
            mcts = MCTS(
                env=env.mathy,
                model=self.local_model,
                num_mcts_sims=self.args.mcts_sims,
                epsilon=epsilon,
            )
        selector: ActionSelector
        if mcts is not None and self.args.action_strategy == "mcts_e_unreal":
            if (
                self.args.use_reward_prediction is False
                and self.args.use_value_replay is False
            ):
                raise EnvironmentError(
                    "This model is not configured for reward prediction/value replay. "
                    "To use UNREAL MCTS you must enable at least one UNREAL aux task."
                )

            selector = UnrealMCTSActionSelector(
                worker=self,
                mcts=mcts,
                epsilon=self.args.unreal_mcts_epsilon,
                episode=A3CWorker.global_episode,
            )
        elif mcts is not None and self.args.action_strategy == "mcts":
            selector = MCTSActionSelector(
                worker=self, mcts=mcts, episode=A3CWorker.global_episode
            )
        elif mcts is not None and self.args.action_strategy == "mcts_worker_0":
            if self.worker_idx == 0:
                selector = MCTSActionSelector(
                    worker=self, mcts=mcts, episode=A3CWorker.global_episode
                )
            else:
                selector = A3CEpsilonGreedyActionSelector(
                    worker=self,
                    epsilon=self.greedy_epsilon,
                    episode=A3CWorker.global_episode,
                )
        elif mcts is not None and self.args.action_strategy == "mcts_worker_n":
            if self.worker_idx != 0:
                selector = MCTSActionSelector(
                    worker=self, mcts=mcts, episode=A3CWorker.global_episode
                )
            else:
                selector = A3CEpsilonGreedyActionSelector(
                    worker=self,
                    epsilon=self.greedy_epsilon,
                    episode=A3CWorker.global_episode,
                )
        elif mcts is not None and self.args.action_strategy == "mcts_recover":
            selector = MCTSRecoveryActionSelector(
                worker=self,
                mcts=mcts,
                episode=A3CWorker.global_episode,
                recover_threshold=self.args.mcts_recover_time_threshold,
                base_selector=A3CEpsilonGreedyActionSelector(
                    worker=self,
                    epsilon=self.greedy_epsilon,
                    episode=A3CWorker.global_episode,
                ),
            )
        elif self.args.action_strategy in ["a3c", "unreal"]:
            selector = A3CEpsilonGreedyActionSelector(
                worker=self,
                epsilon=self.greedy_epsilon,
                episode=A3CWorker.global_episode,
            )
        elif self.args.action_strategy == "a3c-eval":
            selector = A3CGreedyActionSelector(
                worker=self, episode=A3CWorker.global_episode
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
        last_state: MathyObservation = env.reset(rnn_size=self.args.lstm_units)
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = -1

        selector = self.build_episode_selector(env)

        # Set RNN to 0 state for start of episode
        self.local_model.embedding.reset_rnn_state()

        # Start with the "init" sequence
        rnn_state_h = self.local_model.embedding.state_h.numpy()
        rnn_state_c = self.local_model.embedding.state_c.numpy()
        seq_start = env.start_observation([rnn_state_h, rnn_state_c])
        self.local_model.predict_next(
            observations_to_window([seq_start, last_state])
        )

        first_step = True
        while not done and A3CWorker.request_quit is False:
            if self.args.print_training and self.worker_idx == 0:
                env.render("terminal")
            # store rnn state for replay training
            rnn_state_h = self.local_model.embedding.state_h.numpy()
            rnn_state_c = self.local_model.embedding.state_c.numpy()
            last_rnn_state = [rnn_state_h, rnn_state_c]
            # named tuples are read-only, so add rnn state to a new copy
            last_state = MathyObservation(
                nodes=last_state.nodes,
                mask=last_state.mask,
                hints=last_state.hints,
                type=last_state.type,
                time=last_state.time,
                rnn_state=last_rnn_state,
            )

            action = selector.select(
                last_state=env.state,
                last_observation=last_state,
                last_action=last_action,
                last_reward=last_reward,
                last_rnn_state=last_rnn_state,
            )
            # store rnn state for replay training
            rnn_state_h = self.local_model.embedding.state_h.numpy()
            rnn_state_c = self.local_model.embedding.state_c.numpy()

            # Take an env step
            new_state, reward, done, _ = env.step(action)
            new_state = MathyObservation(
                nodes=new_state.nodes,
                mask=new_state.mask,
                hints=new_state.hints,
                type=new_state.type,
                time=new_state.time,
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
            episode_memory.store(last_state, action, reward, grouping_change, frame)
            if time_count == self.args.update_freq or done:
                keep_experience = bool(
                    self.args.use_value_replay or self.args.use_reward_prediction
                )
                if self.args.action_strategy == "unreal":
                    keep_experience = True
                elif self.args.action_strategy == "mcts_e_unreal":
                    unreal = cast(UnrealMCTSActionSelector, selector)
                    keep_experience = unreal.use_mcts or not self.experience.is_full()
                self.update_global_network(
                    done, new_state, episode_memory, keep_experience=keep_experience
                )
                self.maybe_write_histograms()
                time_count = 0
                if done:
                    if self.args.print_training and self.worker_idx == 0:
                        env.render("terminal")
                    self.finish_episode(ep_reward, ep_steps, env.state)

            ep_steps += 1
            time_count += 1
            last_state = new_state
            last_action = action
            last_reward = reward
            first_step = False

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
                f"rewards/mean_episode_reward",
                data=A3CWorker.global_moving_average_reward,
                step=step,
            )

    def maybe_write_histograms(self):
        if self.worker_idx != 0:
            return
        step = self.global_model.global_step.numpy()
        summary_interval = 50
        next_write = self.last_histogram_write + summary_interval
        if step >= next_write or self.last_histogram_write == -1:
            with self.writer.as_default():
                self.last_histogram_write = step
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.global_step
                    )

    def update_global_network(
        self,
        done: bool,
        new_state: MathyObservation,
        episode_memory: EpisodeMemory,
        keep_experience: bool,
    ):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            loss_tuple = self.compute_loss(
                done=done,
                new_state=new_state,
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
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
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

    def loss_normalize(self, loss, epsilon=1e-5):
        """Loss normalization trick from Hvass Labs:
        https://github.com/tensorflow/tensorflow/issues/6414#issuecomment-286118241
        """
        update_condition: bool = self.global_model.global_step.numpy() % 10 == 0
        # Variable used for storing the scalar-value of the loss-function.
        loss_value = tf.Variable(1.0)

        # Expression used for either updating the scalar-value or
        # just re-using the old value.
        # Note that when loss_value.assign(loss) is evaluated, it
        # first evaluates the loss-function which is a TensorFlow
        # expression, and then assigns the resulting scalar-value to
        # the loss_value variable.
        loss_value_updated = tf.cond(
            update_condition, lambda: loss_value.assign(loss), lambda: loss_value
        )

        # Expression for the normalized loss-function.
        loss_normalized = loss / (loss_value_updated + epsilon)

        return loss_normalized

    def soften_policy_targets(self, hard_targets, observations: List[MathyObservation]):
        policy = np.array(hard_targets[:], dtype="float32")
        for i, timestep in enumerate(policy):
            observation = observations[i]
            # The policy coming from the network will usually already include weight
            # for each active rule, but we want the model to be less sure of itself
            # with the targets so we adjust the policy weightings slightly.
            policy_mask = np.array(observation.mask[:], dtype="float32")
            # assert numpy.count_nonzero(policy_mask) == numpy.count_nonzero(policy)
            # The policy mask has 0.0 and 1.0 values. We scale the mask down so that
            # 1 becomes smaller, and then we elementwise add the policy and the mask to
            # increment each valid action by the scaled value
            policy_soft = timestep + (policy_mask * 0.01)
            # Finally we re-normalize the values so that they sum to 1.0 like they
            # did before we incremented them.
            policy_soft /= np.sum(policy_soft)
            policy[i] = policy_soft
        return tf.convert_to_tensor(policy)

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
        # action_values = [r.reward for r in episode_memory.frames]
        # actions = tf.expand_dims(episode_memory.actions, axis=1)
        # pi_loss = discrete_policy_gradient_loss(trimmed_logits, actions, action_values)

        # Advantage is the difference between the final calculated discount
        # rewards, and the current Value function prediction of the rewards
        advantage = discounted_rewards - values

        # Value loss
        value_loss = advantage ** 2
        value_loss = tf.reduce_mean(value_loss)

        # Policy Loss

        action_targets = tf.one_hot(
            episode_memory.actions, depth=self.action_size * sequence_length
        )
        # NOTE: only because we need to do batch-length trimming on the ops inside
        soften = False
        if soften:
            action_targets = self.soften_policy_targets(
                action_targets, episode_memory.observations
            )
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=action_targets, logits=policy_logits
            )
        else:
            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=episode_memory.actions, logits=policy_logits
            )

        policy_loss *= tf.stop_gradient(advantage)
        policy_loss = tf.reduce_mean(policy_loss)
        # Scale the policy loss down by the sequence length to avoid exploding losses
        policy_loss /= sequence_length

        total_loss = value_loss * 0.5 + policy_loss + entropy_loss
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/policy_loss", data=policy_loss, step=step
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/value_loss", data=value_loss, step=step
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/losses/entropy_loss",
            data=entropy_loss,
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/advantage",
            data=tf.reduce_mean(advantage),
            step=step,
        )
        tf.summary.scalar(
            f"worker_{self.worker_idx}/entropy",
            data=tf.reduce_mean(h_loss.extra.entropy),
            step=step,
        )

        return (policy_loss, value_loss, entropy_loss, total_loss, discounted_rewards)

    def compute_grouping_change_loss(
        self, done, new_state, episode_memory: EpisodeMemory, clip: bool = True
    ):
        change_signals = [signal for signal in episode_memory.grouping_changes]
        signals_tensor = tf.convert_to_tensor(change_signals)
        loss = tf.reduce_mean(signals_tensor)
        if clip is True:
            loss = tf.clip_by_value(loss, -1.0, 1.0)
        return loss

    def rp_samples(self) -> Tuple[MathyWindowObservation, float]:
        output = MathyWindowObservation(
            nodes=[], mask=[], hints=[], type=[], rnn_state=[[], []], time=[]
        )
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
        max_steps = 5
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

    def compute_value_replay_loss(self, done, new_state, episode_memory: EpisodeMemory):
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
        new_state: MathyObservation,
        episode_memory: EpisodeMemory,
        keep_experience=False,
        gamma=0.99,
    ):
        with self.writer.as_default():
            step = self.global_model.global_step
            loss_tuple = self.compute_policy_value_loss(done, new_state, episode_memory)
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
                    done, new_state, episode_memory
                )
                gc_loss *= aux_weight
                total_loss += gc_loss
                aux_losses["gc"] = gc_loss
            if self.experience.is_full():
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


class ActionSelector:
    """An episode-specific stategy action using some user-defined method"""

    def __init__(self, *, worker: A3CWorker, episode: int):
        self.worker = worker
        self.episode = episode

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:
        return 0


class A3CGreedyActionSelector(ActionSelector):
    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:
        probs, _ = self.worker.local_model.predict_next(
            observations_to_window([last_observation])
        )
        return np.argmax(probs)


class A3CEpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, *, epsilon: float, **kwargs):
        super(A3CEpsilonGreedyActionSelector, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.noise_alpha = 0.3
        self.use_noise = False

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:

        probs, _ = self.worker.local_model.predict_next(
            observations_to_window([last_observation])
        )
        # Apply dirichlet noise to the root node (ala MCTS search)
        # TODO: Note sure if this is useful without the follow up tree search
        if self.use_noise is True:
            noise = np.random.dirichlet([self.noise_alpha] * len(probs))
            noise = noise * np.array(last_observation.mask)
            probs += noise
            pi_sum = np.sum(probs)
            if pi_sum > 0:
                probs /= pi_sum

            self.first_step = False

        no_random = (
            not self.worker.args.main_worker_use_epsilon and self.worker.worker_idx == 0
        )
        if not no_random and np.random.random() < self.epsilon:
            # Select a random action
            action_mask = last_observation.mask[:]
            # normalize all valid action to equal probability
            actions = action_mask / np.sum(action_mask)
            action = np.random.choice(len(actions), p=actions)
        else:
            action = np.argmax(probs)
        return action


class MCTSActionSelector(ActionSelector):
    def __init__(self, mcts: MCTS, **kwargs):
        super(MCTSActionSelector, self).__init__(**kwargs)
        self.mcts = mcts

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:
        probs = self.mcts.getActionProb(last_state, last_rnn_state)
        return np.argmax(probs)


class MCTSRecoveryActionSelector(MCTSActionSelector):
    def __init__(self, base_selector: ActionSelector, recover_threshold=0.80, **kwargs):
        super(MCTSRecoveryActionSelector, self).__init__(**kwargs)
        self.base_selector = base_selector
        self.recover_threshold = recover_threshold

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:
        if last_observation.time[0] >= self.recover_threshold:
            probs = self.mcts.getActionProb(last_state, last_rnn_state)
            return np.argmax(probs)
        return self.base_selector.select(
            last_state=last_state,
            last_observation=last_observation,
            last_action=last_action,
            last_reward=last_reward,
            last_rnn_state=last_rnn_state,
        )


class UnrealMCTSActionSelector(A3CEpsilonGreedyActionSelector):
    def __init__(self, *, mcts: MCTS, **kwargs):
        super(UnrealMCTSActionSelector, self).__init__(**kwargs)
        self.mcts = mcts
        can_mcts = self.worker.worker_idx != 0
        self.use_mcts = can_mcts and bool(np.random.random() < self.epsilon)

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_observation: MathyObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> int:
        if self.use_mcts is False:
            return super(UnrealMCTSActionSelector, self).select(
                last_state=last_state,
                last_observation=last_observation,
                last_action=last_action,
                last_reward=last_reward,
                last_rnn_state=last_rnn_state,
            )
        probs = self.mcts.getActionProb(last_state, last_rnn_state)
        return np.argmax(probs)
