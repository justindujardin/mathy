# coding: utf8
import json
import os
import random
import time
from datetime import timedelta

import numpy
import plac
import tensorflow as tf
from colr import color

from mathy.agent.controller import MathModel
from mathy.agent.curriculum.level1 import lessons
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.lesson_runner import lesson_runner
from mathy.agent.training.math_experience import (
    MathExperience,
    balanced_reward_experience_samples,
)
from mathy.agent.training.mcts import MCTS
from mathy.agent.training.practice_runner import (
    ParallelPracticeRunner,
    PracticeRunner,
    RunnerConfig,
)
from mathy.agent.training.practice_session import PracticeSession
from mathy.mathy_env import MathyEnv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    transfer_from=(
        "The name of another model to warm start this one from. Think Transfer Learning",
        "positional",
        None,
        str,
    ),
    lesson_id=("The lesson plan to execute by ID", "option", "l", str),
    learning_rate=("The learning rate to use when training", "option", "lr", float),
    initial_train=(
        "When true, train the network on everything in `examples.json` in the checkpoint directory",
        "flag",
        "t",
    ),
    verbose=(
        "When true, print all problem moves rather than just during evaluation",
        "flag",
        "v",
    ),
)
def main(
    model_dir,
    transfer_from=None,
    lesson_id=None,
    initial_train=False,
    verbose=False,
    learning_rate=3e-4,
    parallel=True,
):
    global lessons
    shuffle_lessons = False
    min_train_experience = 256
    eval_interval = 2
    short_term_size = 768
    long_term_size = 8192
    initial_train_iterations = 10
    episode_counter = 0
    counter = 0
    training_epochs = 8
    controller = MathyEnv(verbose=True)
    BaseEpisodeRunner = PracticeRunner if not parallel else ParallelPracticeRunner
    if lesson_id is None:
        plan = lessons[list(lessons)[0]]
    elif lesson_id not in lessons:
        raise ValueError(
            f"[lesson] ERROR: '{lesson_id}' not found in ids. Valid lessons are: {', '.join(lessons)} "
        )
    else:
        plan = lessons[lesson_id]

    lesson = plan.lessons[0]

    class LessonRunner(BaseEpisodeRunner):
        def get_game(self):
            return MathyEnv(verbose=dev_mode, lesson=lesson, max_moves=lesson.max_turns)

        def get_predictor(self, game, all_memory=False):
            return MathModel(game, model_dir, all_memory)

    def session_done(lesson_exercise, num_solved, num_failed):
        return False

    print("Practicing {} - {}...".format(plan.name, lesson.name))
    args = {"self_play_iterations": lesson.problem_count}
    config = RunnerConfig(
        model_dir=model_dir,
        num_mcts_sims=lesson.mcts_sims,
        num_exploration_moves=lesson.num_exploration_moves,
        cpuct=1.0,
    )
    runner = LessonRunner(config)
    c = PracticeSession(runner, lesson)
    c.learn(session_done)

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
