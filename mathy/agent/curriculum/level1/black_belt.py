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

black_belt = build_lesson_plan(
    "black_belt",
    [
        LessonExercise(
            lesson_name="move_then_simplify",
            problem_fn=lambda: combine_terms_after_commuting(15, 20),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=100,
            num_observations=96,
        ),
        LessonExercise(
            lesson_name="twenty_four_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(24),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="commute_challenge",
            problem_fn=lambda: move_around_blockers_two(8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            max_turns=3,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="simplify_in_place",
            problem_fn=lambda: combine_terms_in_place(15, 20, False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            max_turns=3,
            num_observations=64,
        ),
    ],
)

black_belt_practice = build_lesson_plan(
    "black_belt_practice",
    [
        LessonExercise(
            lesson_name="move_then_simplify",
            problem_fn=lambda: combine_terms_after_commuting(15, 20),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=500,
            num_observations=96,
        ),
        LessonExercise(
            lesson_name="twenty_four_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(24),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="commute_challenge",
            problem_fn=lambda: move_around_blockers_two(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="simplify_in_place",
            problem_fn=lambda: combine_terms_in_place(5, 10, False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=64,
        ),
    ],
)
