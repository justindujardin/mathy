import os
from queue import Queue
from typing import Optional
from colr import color

import gym
import numpy as np
import tensorflow as tf

from ..mathy_env_state import MathyEnvState
from .actor_critic_model import ActorCriticModel
from .config import A3CArgs
from .random_agent import RandomAgent
from .worker import A3CWorker
from ..teacher import Teacher, Student, Topic


class A3CAgent:

    args: A3CArgs

    def __init__(self, args: A3CArgs, init_model: Optional[str] = None):
        self.args = args
        print(f"Agent: {os.path.join(args.model_dir, args.model_name)}")
        env = gym.make(self.args.env_name)
        self.action_size = env.action_space.n
        self.teacher = Teacher(
            topic_names=self.args.topics, num_students=self.args.num_workers
        )

        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, "tensorboard")
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        self.global_model = ActorCriticModel(
            args=args, predictions=self.action_size, optimizer=self.optimizer
        )
        # Initialize the global model with a random observation
        self.global_model.maybe_load(env.reset(), do_init=True)

    def train(self):
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.env_name, self.args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()
        workers = [
            A3CWorker(
                global_model=self.global_model,
                action_size=self.action_size,
                args=self.args,
                teacher=self.teacher,
                # args=self.args.copy(update={"env_name": many_workers(i)}),
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
        obs = state.to_input_features(env.action_space.mask, return_batch=True)
        policy, value, masked_policy = self.global_model.call_masked(
            obs, env.action_space.mask
        )
        policy = tf.nn.softmax(masked_policy)
        action = np.argmax(policy)
        return action

    def play(self, env=None, loop=False):
        if env is None:
            env = gym.make(self.args.env_name).unwrapped
        model = self.global_model
        model.maybe_load(env.reset())
        if self.args.algorithm == "random":
            random_agent = RandomAgent(self.args.env_name, self.args.max_eps)
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
                    policy, value, probs = model.call_masked(state)
                    action = np.random.choice(len(probs), p=probs)
                    # NOTE: performance on greedy is terrible. Acting according
                    #       to policy probs solves many more problems.
                    # action = np.argmax(probs)
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
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()
