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
observations = 128

green_belt_practice = build_lesson_plan(
    "green_belt_practice",
    [
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="eight_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="move_then_simplify_1",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=4,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="combine_in_place_1",
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=3,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="commute_blockers_1_7",
            problem_fn=lambda: move_around_blockers_one(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="ten_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_7",
            problem_fn=lambda: move_around_blockers_two(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=observations,
        ),
    ],
)


green_belt = build_lesson_plan(
    "green_belt",
    [
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="eight_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(8, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="move_then_simplify_1",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            max_turns=4,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="combine_in_place_1",
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=3,
            mcts_sims=50,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="commute_blockers_1_7",
            problem_fn=lambda: move_around_blockers_one(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="ten_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=observations,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_7",
            problem_fn=lambda: move_around_blockers_two(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=observations,
        ),
    ],
)
