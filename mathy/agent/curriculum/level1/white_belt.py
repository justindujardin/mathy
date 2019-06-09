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

moves_per_complexity = 3
white_belt_observations = 128
white_belt = build_lesson_plan(
    "white_belt",
    [
        LessonExercise(
            lesson_name="three_terms",
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="six_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
    ],
)

white_belt_practice = build_lesson_plan(
    "white_belt_practice",
    [
        LessonExercise(
            lesson_name="three_terms",
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="six_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=white_belt_observations,
            moves_per_complexity=moves_per_complexity,
        ),
    ],
)

