import multiprocessing
import os
from queue import Queue

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym.wrappers import FlattenDictWrapper

from ..agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_NODE_COUNT,
)
from .a3c_worker import A3CWorker
from .actor_critic_model import ActorCriticModel
from .random_agent import RandomAgent
from . import game_for_worker_index


class A3CAgent:
    def __init__(self, args, units=512, model_name="mathy-poly"):
        self.args = args
        self.units = units
        self.model_name = model_name
        self.game_name = "mathy-poly-v0"
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        env = gym.make(self.game_name)
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
            load_model=self.model_name,
        )
        # Initialize the global model with a random observation
        self.global_model.maybe_load(env.reset())

    def train(self):
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.game_name, self.args.max_eps)
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
                game_name=game_for_worker_index(i),
                save_dir=self.save_dir,
                model_name=self.model_name,
                args=self.args,
                shared_layers=[self.shared_network],
                shared_units=self.units,
            )
            for i in range(num_workers)
        ]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
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
        plt.savefig(os.path.join(self.save_dir, f"{self.game_name} Moving Average.png"))
        plt.show()

    def play(self, loop=False):
        env = gym.make(self.game_name, difficulty=4).unwrapped
        model = self.global_model
        model.maybe_load(env.reset())
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
