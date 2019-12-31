import json
import os
from queue import Queue
from typing import List

import gym
import numpy as np

from ...teacher import Teacher
from ..policy_value_model import get_or_create_policy_model, PolicyValueModel
from .config import A3CConfig
from .worker import A3CWorker


class A3CAgent:

    args: A3CConfig
    global_model: PolicyValueModel

    def __init__(self, args: A3CConfig):
        import tensorflow as tf

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
        init_window = env.initial_window(self.args.lstm_units)
        self.global_model = get_or_create_policy_model(
            args=args, env_actions=self.action_size, is_main=True,
        )
        with self.writer.as_default():
            tf.summary.trace_on(graph=True)
            inputs = init_window.to_inputs()
            self.global_model.call_graph(inputs)
            tf.summary.trace_export(
                name="PolicyValueModel", step=0, profiler_outdir=self.log_dir
            )
            tf.summary.trace_off()
            if self.args.verbose:
                print(self.global_model.summary())

    def train(self):
        res_queue = Queue()
        exp_out_queue = Queue()
        A3CWorker.global_episode = 0
        worker_exploration_epsilons = np.geomspace(
            self.args.e_greedy_min, self.args.e_greedy_max, self.args.num_workers
        )
        workers = [
            A3CWorker(
                global_model=self.global_model,
                action_size=self.action_size,
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
                reward = res_queue.get()
                if reward is None:
                    break
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
            A3CWorker.request_quit = True

        # Do an optimistic save incase there's a problem joining the workers
        self.global_model.save()
        [w.join() for w in workers]
        # Do a final save after joining to get the very latest model
        self.global_model.save()
        print("Done. Bye!")
