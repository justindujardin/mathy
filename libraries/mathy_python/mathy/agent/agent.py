import json
import os
from queue import Queue
from typing import List

import gym
import numpy as np

from ..teacher import Teacher
from .model import get_or_create_agent_model, AgentModel
from .config import AgentConfig
from .worker import A3CWorker
from ..envs.gym import MathyGymEnv


class A3CAgent:

    args: AgentConfig
    global_model: AgentModel

    def __init__(self, args: AgentConfig, env_extra: dict = None):
        import tensorflow as tf

        self.args = args
        self.env_extra = env_extra if env_extra is not None else {}
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
        env: MathyGymEnv = gym.make(self.teacher.get_env(0, 0), **self.env_extra)
        self.action_size = env.action_space.n
        self.log_dir = os.path.join(self.args.model_dir, "tensorboard")
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.global_model = get_or_create_agent_model(
            config=args, predictions=self.action_size, is_main=True, env=env.mathy
        )
        if self.args.verbose:
            print(self.global_model.summary())

    def train(self):
        res_queue = Queue()
        A3CWorker.global_episode = 0
        worker_exploration_epsilons = np.geomspace(
            self.args.e_greedy_min, self.args.e_greedy_max, self.args.num_workers
        )
        workers = [
            A3CWorker(
                env_extra=self.env_extra,
                global_model=self.global_model,
                action_size=self.action_size,
                greedy_epsilon=worker_exploration_epsilons[i],
                args=self.args,
                teacher=self.teacher,
                worker_idx=i,
                optimizer=self.global_model.optimizer,
                writer=self.writer,
            )
            for i in range(self.args.num_workers)
        ]

        for worker in workers:
            worker.start()

        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
            A3CWorker.request_quit = True

        # Do an optimistic save incase there's a problem joining the workers
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.global_model.save(model_path)
        [w.join() for w in workers]
        # Do a final save after joining to get the very latest model
        self.global_model.save(model_path)
        print("Done. Bye!")

