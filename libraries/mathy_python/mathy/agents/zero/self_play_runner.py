import json
import time
from pathlib import Path

import numpy as np

from mathy.core.parser import ExpressionParser, ParserException

from ...env import MathyEnv
from .config import SelfPlayConfig
from .lib.average_meter import AverageMeter
from .lib.progress.bar import Bar
from .practice_runner import ParallelPracticeRunner, PracticeRunner
from .practice_session import PracticeSession


def self_play_runner(config: SelfPlayConfig):
    """Practice a concept for up to (n) lessons or until the concept is learned as defined
    by the lesson plan. """
    BaseEpisodeRunner = (
        PracticeRunner if config.num_workers < 2 else ParallelPracticeRunner
    )
    if config.profile and config.num_workers > 1:
        raise NotImplementedError("zero agent does not support multiprocess profiling")
    lesson_name = "mathy-poly-easy-v0"
    if config.verbose:
        print(config.json(indent=2))

    class LessonRunner(BaseEpisodeRunner):  # type:ignore
        def get_env(self):
            import gym

            return gym.make(lesson_name)

        def get_model(self, game):
            from ...agents.policy_value_model import (
                get_or_create_policy_model,
                PolicyValueModel,
            )

            model: PolicyValueModel = get_or_create_policy_model(
                args=config, predictions=game.action_space.n, is_main=True
            )
            return model

    print("Practicing {}...".format(config.topics))
    runner = LessonRunner(config)
    c = PracticeSession(runner, config, env_name=lesson_name)
    c.learn()
