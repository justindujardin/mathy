# coding: utf8
from mathzero.training.lessons import LessonExercise, LessonPlan
from mathzero.training.lessons import LessonExercise, build_lesson_plan
from mathzero.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    simplify_multiple_terms,
    rand_var,
    maybe_int,
    get_rand_vars,
)
import random

moves_per_complexity = 6


def split_in_two_random(max_items: int):
    count = 0
    while True:
        count += 1
        factor = random.uniform(0, 1)
        left = int(factor * max_items)
        right = max_items - left
        if left + right == max_items:
            return left, right
        if count > 100:
            break
    raise ValueError(
        "something is wrong. failed to generate two numbers with a sum 100 times"
    )


def combine_like_terms_complexity_challenge(easy=True):
    # two max move problems with large complexity (i.e. 45 terms) with two terms side-by-side.
    # the idea is that if you do a bunch of these, the model will learn that DF/CA is a good
    # combination regardless of the position.
    # NOTE: This could also be an AWESOME way to do really large problems without them taking forever.
    #
    # Example:
    #  "4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x)  + 43n + 17j"
    #  max_turns = 2 = Distributive factor + Constant Arithmetic

    total_terms = random.randint(16, 26)
    var = rand_var()
    focus_chunk = f"{maybe_int()}{var} + {maybe_int()}{var}"
    if easy:
        focus_chunk = f"({focus_chunk})"
    num_noise_terms = total_terms - 2
    noise_vars = get_rand_vars(num_noise_terms, [var])

    out_terms = []

    left_num, right_num = split_in_two_random(num_noise_terms)
    for i in range(left_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}")
    out_terms.append(focus_chunk)
    for i in range(right_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}")

    complexity = total_terms
    problem = " + ".join(out_terms)
    return problem, complexity


def get_blocker(num_blockers=1, exclude_vars=[]):
    """Get a string of terms to place between target simplification terms
    in order to challenge the agent's ability to use commutative/associative
    rules to move terms around."""
    vars = get_rand_vars(num_blockers, exclude_vars)
    out_terms = []
    for i in range(num_blockers):
        out_terms.append("{}{}".format(maybe_int(), vars[i]))
    return " + ".join(out_terms)


def move_around_blockers_one(number_blockers):
    # two like terms separated by (n) blocker terms, e.g. 2 ~ "4x + (y + f) + x"
    var = rand_var()
    complexity = 2 + number_blockers
    blockers = get_blocker(number_blockers, [var])
    problem = "{}{} + {} + {}{}".format(maybe_int(), var, blockers, maybe_int(), var)
    return problem, complexity


def move_around_blockers_two(number_blockers):
    # two like terms with three blockers: "7a + 4x + (2f + j) + x + 3d"
    rand_vars = get_rand_vars(3)
    [one_var, two_var, three_var] = rand_vars
    complexity = 4 + number_blockers
    problem = "{}{} + {}{} + {} + {}{} + {}{}".format(
        maybe_int(),
        one_var,
        maybe_int(),
        two_var,
        get_blocker(number_blockers, rand_vars),
        maybe_int(),
        two_var,
        maybe_int(),
        three_var,
    )
    return problem, complexity


def move_around_interleaved_like_terms(number_terms, number_pairs):
    # interleaved multiple like variables: "4x + 2y + 6x + 3y"
    complexity = number_terms * number_pairs
    terms = []
    rand_vars = get_rand_vars(number_terms)
    for i in range(number_pairs):
        for j in range(number_terms):
            terms.append("{}{}".format(maybe_int(), rand_vars[j]))
    return " + ".join(terms), complexity


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


yellow_belt = build_lesson_plan(
    "yellow_belt",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_blockers_1",
            problem_fn=lambda: move_around_blockers_one(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
    ],
)


combine_forced = build_lesson_plan(
    "combine_terms_forced",
    [
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=4,
            problem_fn=lambda: combine_like_terms_complexity_challenge(),
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
        LessonExercise(
            lesson_name="two_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=4,
            problem_fn=lambda: combine_like_terms_complexity_challenge(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            max_turns=3,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="inner_blockers_difficult",
            problem_fn=lambda: move_around_blockers_one(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=250,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=64,
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
            num_observations=64,
        ),
        # LessonExercise(
        #     lesson_name="fourteen_terms",
        #     problem_count=1,
        #     problem_fn=lambda: simplify_multiple_terms(14),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=250,
        #     num_observations=256,
        # ),
        # LessonExercise(
        #     lesson_name="twelve_terms",
        #     problem_count=1,
        #     problem_fn=lambda: simplify_multiple_terms(12),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=500,
        #     num_observations=256,
        # ),
        # LessonExercise(
        #     lesson_name="ten_terms",
        #     problem_count=1,
        #     problem_fn=lambda: simplify_multiple_terms(10),
        #     problem_type=MODE_SIMPLIFY_POLYNOMIAL,
        #     mcts_sims=250,
        #     num_observations=256,
        # ),
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
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
        ),
    ],
)
