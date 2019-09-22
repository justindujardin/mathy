import math
import os
import threading
import time
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from colr import color

from trfl import discrete_policy_entropy_loss, discrete_policy_gradient_loss

from ..core.expressions import MathTypeKeysMax
from ..features import (
    FEATURE_FWD_VECTORS,
    FEATURE_MOVE_MASK,
    calculate_grouping_control_signal,
)
from ..state import MathyEnvState
from ..teacher import Student, Teacher, Topic
from ..util import GameRewards
from .config import MathyArgs
from .episode_memory import EpisodeMemory
from .experience import Experience, ExperienceFrame
from .model import MathyModel
from .util import MPClass, record


class MathyActor(MPClass):
    """Actors gather experience and submit it to the learner for replay training"""

    args: MathyArgs
    request_quit = False
    envs: Dict[str, Any]

    def __init__(
        self,
        args: MathyArgs,
        result_queue: Queue,
        command_queue: Queue,
        experience: Experience,
        worker_idx: int,
        greedy_epsilon: float,
        writer: tf.summary.SummaryWriter,
        teacher: Teacher,
    ):
        super(MathyActor, self).__init__()
        self.args = args
        self.iteration = 0
        self.experience = experience
        self.greedy_epsilon = greedy_epsilon
        self.worker_step_count = 0
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.worker_idx = worker_idx
        self.teacher = teacher
        self.envs = {}
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)
        self.envs[env_name] = gym.make(env_name)
        self.action_size = self.envs[env_name].action_space.n
        self.writer = writer
        self.model = MathyModel(args=args, predictions=self.action_size)
        self.model.maybe_load(self.envs[env_name].initial_state())
        print(f"[actor{worker_idx}] e:{self.greedy_epsilon} t: {self.args.topics}")

    def run(self):
        if self.args.profile:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        episode_memory = EpisodeMemory(experience_queue=self.result_queue)
        while MathyActor.request_quit is False:
            try:
                ctrl = self.command_queue.get_nowait()
                if ctrl == "load_model":
                    print(f"[Worker{self.worker_idx}] loading latest learner")
                    self.model.maybe_load()
            except BaseException:
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
                    step = self.model.global_step
                    tf.summary.scalar(
                        f"worker_{self.worker_idx}/{student.topic}/success_rate",
                        data=win_pct,
                        step=step,
                    )
                    tf.summary.scalar(
                        f"worker_{self.worker_idx}/{student.topic}/difficulty",
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

    def run_episode(self, episode_memory: EpisodeMemory):
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)
        if env_name not in self.envs:
            self.envs[env_name] = gym.make(env_name)
        env = self.envs[env_name]
        episode_memory.clear()
        self.ep_loss = 0
        ep_reward = 0.0
        ep_steps = 0
        done = False
        last_state = env.reset()
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = -1
        while not done and MathyActor.request_quit is False:
            # store rnn state for replay training
            rnn_state_h = self.model.embedding.state_h.numpy()
            rnn_state_c = self.model.embedding.state_c.numpy()

            sample = episode_memory.get_current_window(last_state, env.state)

            if not self.experience.is_full():
                # Select a random action from the last timestep mask
                action_mask = sample.mask[-1][-1][:]
                # normalize all valid action to equal probability
                actions = action_mask / np.sum(action_mask)
                action = np.random.choice(len(actions), p=actions)
                value = np.random.random()
            elif np.random.random() < self.greedy_epsilon:
                _, value = self.model.predict_next(sample)
                # Select a random action from the last timestep mask
                action_mask = sample.mask[-1][-1][:]
                # normalize all valid action to equal probability
                actions = action_mask / np.sum(action_mask)
                action = np.random.choice(len(actions), p=actions)
            else:
                probs, value = self.model.predict_next(sample)
                # action = np.random.choice(len(probs), p=probs)
                action = np.argmax(probs)

            # Take an env step
            new_state, reward, done, _ = env.step(action)
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

            if done:
                self.finish_episode(episode_memory, ep_reward, ep_steps, env.state)

            self.worker_step_count += 1
            ep_steps += 1
            last_state = new_state
            last_action = action
            last_reward = reward

            # If the experience buffer is ready to be played from, sleep between
            # timesteps so that we can have more actors than processors
            if self.experience.is_full():
                time.sleep(self.args.actor_timestep_wait)
        return ep_reward

    def maybe_write_episode_summaries(
        self, episode_reward: float, episode_steps: int, last_state: MathyEnvState
    ):
        # Track metrics for all workers
        name = self.teacher.get_env(self.worker_idx, self.iteration)
        step = self.model.global_step
        with self.writer.as_default():

            tf.summary.scalar(
                f"rewards/worker_{self.worker_idx}/episodes",
                data=episode_reward,
                step=step,
            )
            tf.summary.scalar(
                f"steps/worker_{self.worker_idx}/ep_steps",
                data=episode_steps,
                step=step,
            )

            # TODO: track per-worker averages and log them
            # tf.summary.scalar(
            #     f"rewards/worker_{self.worker_idx}/mean_episode_reward",
            #     data=episode_reward,
            #     step=step,
            # )

            agent_state = last_state.agent
            p_text = f"{agent_state.history[0].raw} = {agent_state.history[-1].raw}"
            outcome = "SOLVED" if episode_reward > 0.0 else "FAILED"
            out_text = f"{outcome}: {p_text}"
            tf.summary.text(
                f"{name}/worker_{self.worker_idx}/summary", data=out_text, step=step
            )

            if self.worker_idx == 0:
                # Track global model metrics
                tf.summary.scalar(
                    f"rewards/mean_episode_reward",
                    data=MathyActor.global_moving_average_reward,
                    step=step,
                )

    def finish_episode(
        self,
        episode_memory: EpisodeMemory,
        episode_reward: float,
        episode_steps: int,
        last_state: MathyEnvState,
    ):
        env_name = self.teacher.get_env(self.worker_idx, self.iteration)
        reward_sum = 0.0  # terminal
        discounted_rewards: List[float] = []
        for reward in episode_memory.rewards[::-1]:
            reward_sum = reward + self.args.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(
            value=np.array(discounted_rewards)[:, None], dtype=tf.float32
        )
        # Store experience frames now that we have finalized discounted
        # reward values.
        episode_memory.commit_frames(discounted_rewards)
        MathyActor.global_moving_average_reward = record(
            episode_reward,
            self.worker_idx,
            episode_steps,
            env_name,
            self.experience.is_full(),
        )
        self.maybe_write_episode_summaries(episode_reward, episode_steps, last_state)
