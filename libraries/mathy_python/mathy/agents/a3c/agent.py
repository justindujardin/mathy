import json
import os
from queue import Queue
from typing import List

import gym
import numpy as np
import tensorflow as tf
from colr import color

from ...state import MathyEnvState, MathyObservation, observations_to_window
from ...teacher import Teacher
from ..policy_value_model import PolicyValueModel, get_or_create_policy_model
from ..base_config import A3CConfig
from .worker import A3CWorker


class A3CAgent:

    args: A3CConfig

    def __init__(self, args: A3CConfig):
        self.args = args
        if self.args.verbose:
            print(f"Agent: {os.path.join(args.model_dir, args.model_name)}")
            print(f"Config: {json.dumps(self.args.dict(), indent=2)}")
        self.teacher = Teacher(
            topic_names=self.args.topics,
            num_students=self.args.num_workers,
            difficulty=self.args.difficulty,
            eval_window=self.args.teacher_evaluation_steps,
            win_threshold=self.args.teacher_promote_wins,
            lose_threshold=self.args.teacher_demote_wins,
        )
        env = gym.make(self.teacher.get_env(0, 0))
        self.action_size = env.action_space.n
        self.log_dir = os.path.join(self.args.model_dir, "tensorboard")
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.global_model = get_or_create_policy_model(
            args, self.action_size, env.initial_window(self.args.lstm_units)
        )
        with self.writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            self.global_model.call_graph(
                env.initial_window(self.args.lstm_units).to_inputs()
            )
            tf.summary.trace_export(
                name="PolicyValueModel", step=0, profiler_outdir=self.log_dir
            )
            tf.summary.trace_off()
            print(self.global_model.summary())

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
                experience_queue=exp_out_queue,
                cmd_queue=cmd_queues[i],
                greedy_epsilon=worker_exploration_epsilons[i],
                args=self.args,
                teacher=self.teacher,
                worker_idx=i,
                optimizer=self.global_model.optimizer,
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

    def play(self, loop=False):
        model = self.global_model
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
