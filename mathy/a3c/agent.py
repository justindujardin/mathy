import os
from queue import Queue
from typing import Optional, List
from colr import color

import gym
import numpy as np
import tensorflow as tf

from ..state import MathyEnvState, observations_to_window
from .actor_critic_model import ActorCriticModel
from .config import A3CArgs
from .worker import A3CWorker
from ..teacher import Teacher, Student, Topic
from ..r2d2.experience import Experience, ExperienceFrame


class A3CAgent:

    args: A3CArgs

    def __init__(self, args: A3CArgs):
        self.args = args
        if self.args.verbose:
            print(f"Agent: {os.path.join(args.model_dir, args.model_name)}")
            print(f"Config: {self.args.dict()}")
        self.teacher = Teacher(
            topic_names=self.args.topics,
            num_students=self.args.num_workers,
            difficulty=self.args.difficulty,
        )
        env = gym.make(self.teacher.get_env(0, 0))
        self.action_size = env.action_space.n
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, "tensorboard")
        )
        # Clip gradients: https://youtu.be/yCC09vCHzF8?t=3885
        self.optimizer = tf.keras.optimizers.Adam(
            lr=args.lr, clipnorm=1.0, clipvalue=0.5
        )
        self.global_model = ActorCriticModel(
            args=args, predictions=self.action_size, optimizer=self.optimizer
        )
        # Initialize the global model with a random observation
        self.global_model.maybe_load(
            env.initial_window(self.args.lstm_units), do_init=True
        )
        self.optimizer.iterations = self.global_model.global_step
        self.experience = Experience(self.args.history_size, self.args.ready_at)

    def train(self):

        res_queue = Queue()
        exp_out_queue = Queue()
        cmd_queues: List[Queue] = [Queue() for i in range(self.args.num_workers)]
        worker_exploration_epsilons = np.geomspace(
            self.args.e_greedy_min, self.args.e_greedy_max, self.args.num_workers
        )
        workers = [
            A3CWorker(
                global_model=self.global_model,
                action_size=self.action_size,
                exp_out=exp_out_queue,
                cmd_queue=cmd_queues[i],
                greedy_epsilon=worker_exploration_epsilons[i],
                args=self.args,
                teacher=self.teacher,
                worker_idx=i,
                optimizer=self.optimizer,
                result_queue=res_queue,
                writer=self.writer,
            )
            for i in range(self.args.num_workers)
        ]

        for i, worker in enumerate(workers):
            worker.start()

        try:
            while True:
                try:
                    # Share experience between workers
                    index, frames = exp_out_queue.get_nowait()
                    # It's lame, but post it back to the others.
                    for i, q in enumerate(cmd_queues):
                        # Don't post back to self
                        if i == index:
                            continue
                        q.put(("experience", frames))
                except BaseException:
                    pass

                reward = res_queue.get()
                if reward is None:
                    break
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
            A3CWorker.request_quit = True
            self.global_model.save()

        [w.join() for w in workers]
        print("Done. Bye!")

    def choose_action(self, env, state: MathyEnvState):
        obs = state.to_observation(env.action_space.mask)
        policy, value, masked_policy = self.global_model.call_masked(
            observations_to_window([obs])
        )
        policy = tf.nn.softmax(masked_policy)
        action = np.argmax(policy)
        return action

    def play(self, loop=False):
        model = self.global_model
        model.maybe_load()
        envs = {}
        try:
            episode_counter = 0
            while loop is True:
                env_name = self.teacher.get_env(0, episode_counter)
                if env_name not in envs:
                    envs[env_name] = gym.make(env_name).unwrapped
                env = envs[env_name]
                state = env.reset(rnn_size=self.args.lstm_units)
                done = False
                step_counter = 0
                reward_sum = 0
                while not done:
                    env.render(mode="terminal")
                    policy, value, probs = model.call_masked(
                        observations_to_window([state])
                    )
                    action = np.argmax(probs)
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    win = False
                    if done and reward > 0.0:
                        win = True
                        env.render(mode="terminal")
                    if done:
                        print(
                            color(
                                text="SOLVE" if win else "FAIL",
                                fore="green" if win else "red",
                            )
                        )

                    step_counter += 1
                # Episode counter
                episode_counter += 1

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()
