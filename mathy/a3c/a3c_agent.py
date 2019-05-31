import os
import gym
import multiprocessing
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt

import tensorflow as tf

from .actor_critic_model import ActorCriticModel
from .random_agent import RandomAgent
from .a3c_worker import A3CWorker
from .ddqn_agent import DDQNAgent


class A3CAgent:
    def __init__(self, args, units=128):
        self.args = args
        self.game_name = "CartPole-v1"
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        self.shared_network = tf.keras.layers.Dense(
            units, activation="relu", name="shared_network"
        )
        self.global_model = ActorCriticModel(
            self.state_size, self.action_size, shared_layers=[self.shared_network]
        )  # global network
        self.global_model(
            tf.convert_to_tensor(
                value=np.random.random((1, self.state_size)), dtype=tf.float32
            )
        )

    def train(self):
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.game_name, self.args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()

        workers = [
            A3CWorker(
                self.state_size,
                self.action_size,
                self.global_model,
                self.optimizer,
                res_queue,
                i,
                game_name=self.game_name,
                save_dir=self.save_dir,
                args=self.args,
                shared_layers=[self.shared_network],
            )
            for i in range(multiprocessing.cpu_count())
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
        plt.savefig(
            os.path.join(self.save_dir, "{} Moving Average.png".format(self.game_name))
        )
        plt.show()

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(self.save_dir, "model_{}.h5".format(self.game_name))
        print("Loading model from: {}".format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode="rgb_array")
                policy, value = model(
                    tf.convert_to_tensor(value=state[None, :], dtype=tf.float32)
                )
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print(
                    "{}. Reward: {}, action: {}".format(
                        step_counter, reward_sum, action
                    )
                )
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()
