import time
import json
from pathlib import Path
import numpy as np

from mathy.core.parser import ExpressionParser, ParserException
from mathy.envs.gym import MathyGymEnv

from .lib.average_meter import AverageMeter
from .lib.progress.bar import Bar
from ...mathy_env import MathyEnv
from .config import SelfPlayConfig
from .practice_runner import ParallelPracticeRunner, PracticeRunner
from .practice_session import PracticeSession


def self_play_runner(config: SelfPlayConfig):
    """Practice a concept for up to (n) lessons or until the concept is learned as defined
    by the lesson plan. """
    BaseEpisodeRunner = (
        PracticeRunner if config.num_workers < 2 else ParallelPracticeRunner
    )
    lesson_name = "mathy-poly-normal-v0"

    class LessonRunner(BaseEpisodeRunner):  # type:ignore
        def get_game(self):
            import gym

            return gym.make(lesson_name)

        def get_predictor(self, game: MathyGymEnv):
            from ...agents.policy_value_model import PolicyValueModel

            model = PolicyValueModel(args=config, predictions=game.action_space.n)
            model.maybe_load()
            return model

    print("Practicing {}...".format(config.topics))
    runner = LessonRunner(config)
    c = PracticeSession(runner, config, env_name=lesson_name)
    c.learn()


def write_lesson_state(file_path, file_data):
    with Path(file_path).open("w", encoding="utf-8") as f:
        f.write(json.dumps(file_data))
