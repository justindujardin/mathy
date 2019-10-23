import os
from queue import Queue
from typing import List, Optional

import numpy as np
import tensorflow as tf

from ...teacher import Teacher
from ..base_config import BaseConfig
from ..experience import Experience, ExperienceFrame
from .actor import MathyActor
from .learner import MathyLearner


class MathyTrainer:

    args: BaseConfig

    def __init__(self, args: BaseConfig):
        self.args = args
        self.experience = Experience(
            history_size=self.args.history_size, ready_at=self.args.ready_at
        )
        if self.args.verbose:
            print(f"Trainer: {os.path.join(args.model_dir, args.model_name)}")
            print(f"Config: {self.args.dict()}")
        self.teacher = Teacher(
            topic_names=self.args.topics,
            num_students=self.args.num_workers,
            difficulty=self.args.difficulty,
        )
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, "tensorboard")
        )

    def run(self):
        res_queue = Queue()
        cmd_queues: List[Queue] = [Queue() for i in range(self.args.num_workers)]

        all_children = []

        # Create (n) actors for gathering trajectories
        actor_epsilons = np.linspace(0.001, 0.5, self.args.num_workers)
        actors = [
            MathyActor(
                args=self.args,
                command_queue=cmd_queues[i],
                experience=self.experience,
                greedy_epsilon=actor_epsilons[i],
                result_queue=res_queue,
                teacher=self.teacher,
                worker_idx=i,
                writer=self.writer,
            )
            for i in range(self.args.num_workers)
        ]
        all_children += actors

        # Create one learner for training on replay data
        learner = MathyLearner(
            args=self.args,
            command_queues=cmd_queues,
            experience=self.experience,
            writer=self.writer,
        )
        all_children.append(learner)
        for i, worker in enumerate(all_children):
            worker.start()

        try:
            while True:
                experience_frame: Optional[ExperienceFrame] = res_queue.get()
                if experience_frame is not None:
                    self.experience.add_frame(experience_frame)
                else:
                    break
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
            MathyActor.request_quit = True
            MathyLearner.request_quit = True
            learner.model.save()

        [w.join() for w in all_children]
        print("Done. Bye!")
