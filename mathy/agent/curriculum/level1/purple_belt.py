# coding: utf8
import random

from ....game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ...training.lessons import LessonExercise, LessonPlan, build_lesson_plan
from ..problems import (
    get_rand_vars,
    maybe_int,
    maybe_power,
    rand_bool,
    rand_var,
    simplify_multiple_terms,
    combine_terms_after_commuting,
    combine_terms_in_place,
    move_around_blockers_one,
    move_around_blockers_two,
)

moves_per_complexity = 4


purple_sims = 50
purple_belt = build_lesson_plan(
    "purple_belt_practice",
    [
        LessonExercise(
            lesson_name="five_complex_terms",
            problem_fn=lambda: simplify_multiple_terms(5, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="six_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="eight_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="ten_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_sims,
            num_observations=64,
        ),
    ],
)

purple_practice_sims = 250
purple_belt_practice = build_lesson_plan(
    "purple_belt_practice",
    [
        LessonExercise(
            lesson_name="five_complex_terms",
            problem_fn=lambda: simplify_multiple_terms(5, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_practice_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="six_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_practice_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="eight_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_practice_sims,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="ten_complex_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers_proability=0.85, op=None),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=purple_practice_sims,
            num_observations=64,
        ),
    ],
)
