# coding: utf8
"""Executing training and evaluation against the agent curriculum, automatically progressing
to the next level as the agent gets better.
"""
import numpy
import plac
from mathzero.training.lesson_runner import lesson_runner
from mathzero.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    combine_multiple_like_add_terms,
    rand_int,
    rand_var,
)
from mathzero.training.lessons import build_lesson_plan, LessonExercise
import random
import tensorflow as tf


tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity("CRITICAL")


def two_variable_terms():
    variable = rand_var()
    problem = "{}{} + {}{}".format(rand_int(), variable, rand_int(), variable)
    return problem, 2


@plac.annotations(
    agent_name=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    )
)
def main(agent_name="default"):
    lesson_runner(
        agent_name,
        build_lesson_plan(
            "Development Tests",
            [
                LessonExercise(
                    lesson_name="QuickSelfPlay",
                    problem_count=1,
                    problem_fn=lambda: combine_multiple_like_add_terms(3),
                    problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                    max_turns=15,
                    num_exploration_moves=5,
                    mcts_sims=250,
                ),
                LessonExercise(
                    lesson_name="QuickSelfPlay2",
                    problem_count=4,
                    problem_fn=lambda: two_variable_terms(),
                    problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                    num_exploration_moves=10,
                    max_turns=15,
                    mcts_sims=100,
                ),
                LessonExercise(
                    lesson_name="QuickSelfPlay3",
                    problem_count=4,
                    problem_fn=lambda: two_variable_terms(),
                    problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                    num_exploration_moves=10,
                    max_turns=15,
                    mcts_sims=100,
                ),
                LessonExercise(
                    lesson_name="DifficultSelfPlay",
                    problem_count=10,
                    problem_fn=lambda: combine_multiple_like_add_terms(6),
                    problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                    num_exploration_moves=20,
                    max_turns=50,
                    mcts_sims=100,
                ),
            ],
        ),
        parallel=False,
        dev_mode=True,
    )


if __name__ == "__main__":
    plac.call(main)
