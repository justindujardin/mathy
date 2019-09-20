import os
from queue import Queue
from typing import Optional
from colr import color

import gym
import numpy as np
import tensorflow as tf

from ..mathy_env_state import MathyEnvState
from .model import MathyModel
from .config import MathyArgs
from .actor import MathyActor
from .learner import MathyLearner
from .experience import Experience, ExperienceFrame
from ..teacher import Teacher, Student, Topic


class MathyTrainer:

    args: MathyArgs

    def __init__(self, args: MathyArgs):
        self.args = args
        self.experience = Experience(
            history_size=self.args.replay_size, ready_at=self.args.replay_ready
        )
        if self.args.verbose:
            print(f"Trainer: {os.path.join(args.model_dir, args.model_name)}")
            print(f"Config: {self.args.dict()}")
        self.teacher = Teacher(
            topic_names=self.args.topics,
            num_students=self.args.num_actors,
            difficulty=self.args.difficulty,
        )
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, "tensorboard")
        )

    def run(self):
        res_queue = Queue()
        cmd_queue = Queue()

        # Create (n) actors for gathering trajectories
        actors = [
            MathyActor(
                args=self.args,
                teacher=self.teacher,
                experience=self.experience,
                worker_idx=i,
                result_queue=res_queue,
                command_queue=cmd_queue,
                writer=self.writer,
            )
            for i in range(self.args.num_actors)
        ]

        # Create (n) learners for training on replay data
        learner = MathyLearner(
            args=self.args, experience=self.experience, writer=self.writer
        )
        for i, worker in enumerate(actors + [learner]):
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

        [w.join() for w in actors + [learner]]
        print("Done. Bye!")
