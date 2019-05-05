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
white_belt_observations = 128
white_belt = build_lesson_plan(
    "white_belt",
    [
        # LessonExercise(
        #     lesson_name="two_terms",
        #     problem_fn=lambda: simplify_multiple_terms(2),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=100,
        #     num_observations=32,
        # ),
        # LessonExercise(
        #     lesson_name="three_terms",
        #     problem_fn=lambda: simplify_multiple_terms(3),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=100,
        #     num_observations=32,
        # ),
        LessonExercise(
            lesson_name="five_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="move_then_simplify",
            problem_fn=lambda: combine_terms_after_commuting(12, 16),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=4,
            mcts_sims=500,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="simplify_in_place",
            problem_fn=lambda: combine_terms_in_place(12, 16, False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=white_belt_observations,
        ),
        # LessonExercise(
        #     lesson_name="five_terms",
        #     problem_fn=lambda: simplify_multiple_terms(5),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=500,
        #     num_observations=white_belt_observations,
        # ),
        LessonExercise(
            lesson_name="commute_blockers_1_3",
            problem_fn=lambda: move_around_blockers_one(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="five_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(5, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_3",
            problem_fn=lambda: move_around_blockers_two(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
        ),
        LessonExercise(
            lesson_name="inner_blockers_difficult",
            problem_fn=lambda: move_around_blockers_one(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
        ),
    ],
)

white_belt_practice = build_lesson_plan(
    "white_belt_practice",
    [
        LessonExercise(
            lesson_name="commute_grouping_1",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            max_turns=4,
            mcts_sims=500,
            num_observations=128,
        ),
        LessonExercise(
            lesson_name="five_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=128,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            max_turns=3,
            num_observations=128,
        ),
        LessonExercise(
            lesson_name="commute_grouping_2",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=500,
            num_observations=128,
            max_turns=6,
        ),
    ],
)

