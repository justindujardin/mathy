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

observations = 128
green_belt = build_lesson_plan(
    "green_belt",
    [
        LessonExercise(
            lesson_name="eight_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="simplify_in_place_8_9",
            problem_fn=lambda: combine_terms_in_place(8, 9),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=200,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="move_then_simplify_8_9",
            problem_fn=lambda: combine_terms_after_commuting(8, 9),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=200,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="nine_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(9, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=observations,
        ),
    ],
)

green_belt_practice = build_lesson_plan(
    "green_belt_practice",
    [
        LessonExercise(
            lesson_name="eight_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="simplify_in_place_8_9",
            problem_fn=lambda: combine_terms_in_place(8, 9),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=2,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="move_then_simplify_8_9",
            problem_fn=lambda: combine_terms_after_commuting(8, 9),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=3,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="nine_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(9, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
    ],
)
