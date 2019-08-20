import multiprocessing
import os
from queue import Queue
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pydantic import BaseModel

from .actor_critic_model import ActorCriticModel
from .random_agent import RandomAgent
from .worker import A3CWorker


class A3CArgs(BaseModel):
    algorithm: str = "a3c"
    train: bool = False
    lr: float = 3e-4
    update_freq: int = 50
    max_eps: int = 10000
    gamma: float = 0.99
    save_dir: str = "training/a3c/"


class A3CAgent:
    def __init__(
        self, args: A3CArgs, env_name, units=128, init_model: Optional[str] = None
    ):
        self.args = args
        self.units = units
        self.env_name = env_name
        # Model to load weights from when initializing the agent model
        self.init_model = init_model
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        env = gym.make(self.env_name)
        self.action_size = env.action_space.n
        self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        self.shared_network = tf.keras.layers.Dense(
            units, activation="relu", name="shared_network"
        )
        self.global_model = ActorCriticModel(
            units=units,
            predictions=self.action_size,
            shared_layers=[self.shared_network],
            save_dir=self.save_dir,
            load_model=self.env_name,
            init_model=self.init_model,
        )
        # Initialize the global model with a random observation
        self.global_model.maybe_load(env.reset())

    def train(self):
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.env_name, self.args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()
        num_workers = multiprocessing.cpu_count()

        workers = [
            A3CWorker(
                self.action_size,
                self.global_model,
                self.optimizer,
                res_queue,
                i,
                env_name=self.env_name,
                save_dir=self.save_dir,
                args=self.args,
                shared_layers=[self.shared_network],
                shared_units=self.units,
            )
            for i in range(num_workers)
        ]

        for i, worker in enumerate(workers):
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel("Moving average ep reward")
        plt.xlabel("Step")
        plt.savefig(os.path.join(self.save_dir, f"{self.env_name} Moving Average.png"))
        plt.show()

    def choose_action(self, env, state):
        policy, value, masked_policy = self.global_model.call_masked(
            state, env.action_space.mask
        )
        policy = tf.nn.softmax(masked_policy)
        action = np.argmax(policy)
        return action

    def play(self, env=None, loop=False):
        if env is None:
            env = gym.make(self.env_name).unwrapped
        model = self.global_model
        model.maybe_load(env.reset())
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.env_name, self.args.max_eps)
            random_agent.run()
            return
        try:
            while loop is True:
                state = env.reset()
                done = False
                step_counter = 0
                reward_sum = 0
                while not done:
                    env.render(mode="terminal")
                    policy, value, masked_policy = model.call_masked(
                        state, env.action_space.mask
                    )
                    policy = tf.nn.softmax(masked_policy)
                    action = np.argmax(policy)
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    if done and reward > 0.0:
                        env.render(mode="terminal")
                    step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()
