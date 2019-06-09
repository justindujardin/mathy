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


quick_test_plan = build_lesson_plan(
    "dev_test",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=2,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
    ],
)

node_control = build_lesson_plan(
    "node_control",
    [
        LessonExercise(
            lesson_name="move_then_simplify",
            problem_fn=lambda: combine_terms_after_commuting(5, 8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=500,
            num_observations=96,
        ),
        LessonExercise(
            lesson_name="simplify_in_place",
            problem_fn=lambda: combine_terms_in_place(12, 16, False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=64,
        ),
    ],
)


combine_forced = build_lesson_plan(
    "combine_terms_forced",
    [
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            max_turns=2,
            num_observations=128,
        )
    ],
)


lesson_plan = build_lesson_plan(
    "combine_like_terms_1",
    [
        # LessonExercise(
        #     lesson_name="two_terms",
        #     problem_count=4,
        #     problem_fn=lambda: simplify_multiple_terms(2),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=250,
        #     num_observations=64,
        # ),
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack_2",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=512,
        ),
        # LessonExercise(
        #     lesson_name="three_terms",
        #     problem_count=4,
        #     problem_fn=lambda: simplify_multiple_terms(3),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=250,
        #     num_observations=64,
        # ),
        LessonExercise(
            lesson_name="inner_blockers_difficult",
            problem_fn=lambda: move_around_blockers_one(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=250,
            num_observations=512,
        ),
        # LessonExercise(
        #     lesson_name="four_terms",
        #     problem_count=4,
        #     problem_fn=lambda: simplify_multiple_terms(4),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=200,
        #     num_observations=64,
        # ),
        LessonExercise(
            lesson_name="five_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=512,
        ),
        # LessonExercise(
        #     lesson_name="six_terms",
        #     problem_count=1,
        #     problem_fn=lambda: simplify_multiple_terms(6),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=250,
        #     num_observations=256,
        # ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="fourteen_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(14),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="twelve_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(12),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="ten_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(10),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=512,
        ),
    ],
)


lesson_plan_2 = build_lesson_plan(
    "combine_like_terms_2",
    [
        LessonExercise(
            lesson_name="twenty_four_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(24),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack_2",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=512,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=512,
        ),
    ],
)

lesson_plan_3 = build_lesson_plan(
    "combine_like_terms_3",
    [
        LessonExercise(
            lesson_name="needle_in_haystack_3",
            problem_count=4,
            problem_fn=lambda: combine_terms_in_place(easy=False, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=256,
        ),
        LessonExercise(
            lesson_name="three_terms_with_powers",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(3, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=256,
        ),
        LessonExercise(
            lesson_name="seven_terms_with_powers",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7, powers_proability=0.85),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=256,
        ),
    ],
)

lesson_quick = build_lesson_plan(
    "combine_like_terms_1",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            num_observations=1,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            num_observations=1,
            mcts_sims=200,
        ),
    ],
)
