# coding: utf8
from ....game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ...training.lessons import LessonExercise, build_lesson_plan
from ..problems import simplify_multiple_terms


def simple_polynomials(number_terms, sims=500, observations=32):
    return build_lesson_plan(
        "simple_polynomials",
        [
            LessonExercise(
                lesson_name=f"polynomials_{number_terms}_terms",
                problem_fn=lambda: simplify_multiple_terms(number_terms),
                problem_type=MODE_SIMPLIFY_POLYNOMIAL,
                mcts_sims=sims,
                num_observations=observations,
            )
        ],
    )


dev = build_lesson_plan(
    "dev",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=2,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=2,
        ),
    ],
)
