from mathzero.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    combine_multiple_like_add_terms,
    rand_int,
    rand_var,
)
from mathzero.training.lessons import build_lesson_plan, LessonExercise
import random


def two_variable_terms():
    variable = rand_var()
    problem = "{}{} + {}{}".format(rand_int(), variable, rand_int(), variable)
    return problem, 2


lessons = build_lesson_plan(
    "Development Tests",
    [
        LessonExercise(
            lesson_name="QuickSelfPlay",
            problem_count=2,
            problem_fn=lambda: combine_multiple_like_add_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=15,
            mcts_sims=25,
        )
    ],
)
