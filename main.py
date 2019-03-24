# coding: utf8
"""Pre-train the math vectors to improve agent performance"""
import json
import os
import tempfile
from pathlib import Path
import random
from colr import color
import numpy
import plac
import time
from mathzero.training.lessons import LessonExercise, LessonPlan
from mathzero.core.parser import ExpressionParser, ParserException
from mathzero.embeddings.math_game import MathGame
from mathzero.model.math_model import MathModel
from mathzero.training.lessons import LessonExercise, build_lesson_plan
from mathzero.training.practice_runner import (
    ParallelPracticeRunner,
    PracticeRunner,
    RunnerConfig,
)
from mathzero.training.practice_session import PracticeSession
from mathzero.training.problems import MODE_SIMPLIFY_POLYNOMIAL, simplify_multiple_terms
from mathzero.embeddings.math_experience import MathExperience
from mathzero.training.mcts import MCTS
from mathzero.embeddings.actor_mcts import ActorMCTS
from datetime import timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
# tf.compat.v1.logging.set_verbosity("CRITICAL")


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
)
def main(model_dir, transfer_from=None):
    max_moves = 15
    lesson_plan = build_lesson_plan(
        "Embeddings training",
        [
            LessonExercise(
                lesson_name="two terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(2),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=8,
                mcts_sims=250,
                num_exploration_moves=8,
            ),
            LessonExercise(
                lesson_name="three terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(3),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=12,
                num_exploration_moves=12,
                mcts_sims=250,
            ),
            LessonExercise(
                lesson_name="four terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(4),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=16,
                num_exploration_moves=16,
                mcts_sims=250,
            ),
            LessonExercise(
                lesson_name="five terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(5),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=20,
                num_exploration_moves=20,
                mcts_sims=500,
            ),
            LessonExercise(
                lesson_name="six terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(6),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=24,
                num_exploration_moves=24,
                mcts_sims=500,
            ),
            LessonExercise(
                lesson_name="seven terms",
                problem_count=1,
                problem_fn=lambda: simplify_multiple_terms(7),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                max_turns=28,
                num_exploration_moves=28,
                mcts_sims=500,
            ),
            # LessonExercise(
            #     lesson_name="eight terms",
            #     problem_count=1,
            #     problem_fn=lambda: simplify_multiple_terms(8),
            #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            #     max_turns=32,
            #     mcts_sims=500,
            #     num_exploration_moves=32,
            # ),
            # LessonExercise(
            #     lesson_name="nine terms",
            #     problem_count=1,
            #     problem_fn=lambda: simplify_multiple_terms(9),
            #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            #     max_turns=36,
            #     mcts_sims=500,
            #     num_exploration_moves=36,
            # ),
        ],
    )
    counter = 0
    dev_mode = True
    controller = MathGame(verbose=dev_mode, focus_buckets=6)
    experience = MathExperience(model_dir)
    mathy = MathModel(controller.action_size, model_dir, init_model_dir=transfer_from)
    mathy.start()
    while True:
        print("[Lesson:{}]".format(counter))
        counter = counter + 1

        # if counter % eval_interval == 0:
        #     print("Evaluating agent with greedy action tests")

        lessons = lesson_plan.lessons[:]
        while len(lessons) > 0:
            lesson = lessons.pop(0)
            controller.lesson = lesson
            controller.max_moves = lesson.max_turns
            print("\n{} - {}...".format(lesson_plan.name.upper(), lesson.name.upper()))
            env_state, complexity = controller.get_initial_state()
            mcts = MCTS(controller, mathy, 1.0, lesson.mcts_sims)
            actor = ActorMCTS(mcts, lesson.num_exploration_moves)
            final_result = None
            time_steps = []
            start = time.time()
            while final_result is None:
                env_state, train_example, final_result = actor.step(
                    controller, env_state, mathy, time_steps
                )

            elapsed = time.time() - start
            episode_examples, episode_reward, is_win = final_result
            if is_win:
                outcome = "solved"
                fore = "green"
            else:
                outcome = "failed"
                fore = "red"
            print(
                color(
                    " -- duration({}) outcome({})".format(
                        str(timedelta(seconds=elapsed)), outcome
                    ),
                    fore=fore,
                    style="bright",
                )
            )
            experience.add_batch(episode_examples)
            mathy.train(experience.short_term, experience.long_term)
    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
