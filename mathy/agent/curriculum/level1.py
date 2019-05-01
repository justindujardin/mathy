# coding: utf8
from mathy.agent.training.lessons import LessonExercise, LessonPlan
from mathy.agent.training.lessons import LessonExercise, build_lesson_plan
from mathy.agent.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    simplify_multiple_terms,
    rand_var,
    maybe_int,
    maybe_power,
    rand_bool,
    get_rand_vars,
)
import random

moves_per_complexity = 6


def split_in_two_random(value: int):
    """Split a given number into two smaller numbers that sum to it.

    Returns: a tuple of (lower, higher) numbers that sum to the input
    """
    factor = random.uniform(0, 1)
    left = int(factor * value)
    right = value - left
    # always return lower/higher
    return min(left, right), max(left, right)


def combine_terms_in_place(min_terms=16, max_terms=26, easy=True, powers=False):
    """Generate a problem that puts one pair of like terms somewhere inside
    an expression of unlike terms. The agent should be challenged to make its first 
    few moves count when combined with a very small number of maximum moves.

    The hope is that by focusing the agent on selecting the right moves inside of a 
    ridiculously large expression it will learn to select actions to combine like terms
    invariant of the sequence length.
    
    Example:
      "4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x)  + 43n + 17j"
      max_turns=3  actions=[DistributiveFactorOut, ConstantArithmetic]

    NOTE: we usually add one more move than may strictly be necessary to help with 
    exploration where we inject Dirichlet noise in the root tree search node.
    """

    total_terms = random.randint(min_terms, max_terms)
    var = rand_var()
    power_chance = 80 if powers == True else 0
    power = maybe_power(power_chance)
    focus_chunk = f"{maybe_int()}{var}{power} + {maybe_int()}{var}{power}"
    if easy:
        focus_chunk = f"({focus_chunk})"
    num_noise_terms = total_terms - 2
    noise_vars = get_rand_vars(num_noise_terms, [var])

    out_terms = []

    # We take the larger value for the left side to push the terms
    # that have to be matched to the right side of expression. This is
    # so that the model cannot use its existing knowledge about distributive
    # factoring on smaller problems to solve this problem.
    right_num, left_num = split_in_two_random(num_noise_terms)
    for i in range(left_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}{maybe_power(power_chance)}")
    out_terms.append(focus_chunk)
    for i in range(right_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}{maybe_power(power_chance)}")

    complexity = total_terms
    problem = " + ".join(out_terms)
    return problem, complexity


def combine_terms_after_commuting(
    min_terms=5, max_terms=8, commute_blockers=1, easy=True, powers=False
):
    """A problem with a bunch of terms that have no matches, and a single 
    set of two terms that do match, but are separated by one other term.

    The challenge is to commute the terms to each other and simplify in 
    only a few moves. 
    
    Example:  "4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j"
                                             ^-----------^  
    """
    total_terms = random.randint(min_terms, max_terms)
    num_noise_terms = total_terms - 2
    var = rand_var()
    noise_vars = get_rand_vars(num_noise_terms, [var])
    power_chance = 80 if powers == True else 0
    power = maybe_power(power_chance)

    # Build up the blockers to put between the like terms
    blockers = []
    for i in range(commute_blockers):
        current = noise_vars.pop()
        blockers.append(f"{maybe_int()}{current}{maybe_power(power_chance)}")

    focus_chunk = f"{maybe_int()}{var}{power} + {' + '.join(blockers)} + {maybe_int()}{var}{power}"
    # About half of the time focus the agent by grouping the subtree for them
    if rand_bool(50 if easy else 10):
        focus_chunk = f"({focus_chunk})"

    out_terms = []

    # We take the larger value for the left side to push the terms
    # that have to be matched to the right side of expression. This is
    # so that the model cannot use its existing knowledge about distributive
    # factoring on smaller problems to solve this problem.
    right_num, left_num = split_in_two_random(num_noise_terms - commute_blockers)
    for i in range(left_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}{maybe_power(power_chance)}")
    out_terms.append(focus_chunk)
    for i in range(right_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_int()}{current}{maybe_power(power_chance)}")

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
            lesson_name="four_terms",
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_grouping_1",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5,
            mcts_sims=200,
            num_observations=128,
        ),
        LessonExercise(
            lesson_name="commute_blockers_1_3",
            problem_fn=lambda: move_around_blockers_one(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="five_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(5, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_3",
            problem_fn=lambda: move_around_blockers_two(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(6, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_fn=lambda: combine_terms_in_place(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack_2",
            problem_fn=lambda: combine_terms_in_place(False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="inner_blockers_difficult",
            problem_fn=lambda: move_around_blockers_one(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
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
            max_turns=4,
            mcts_sims=500,
            num_observations=64,
        ),
        LessonExercise(
            lesson_name="simplify_in_place",
            problem_fn=lambda: combine_terms_in_place(12, 16, False),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            max_turns=2,
            num_observations=64,
        ),
    ],
)

yellow_belt = build_lesson_plan(
    "yellow_belt",
    [
        LessonExercise(
            lesson_name="commute_blockers_1_3",
            problem_fn=lambda: move_around_blockers_one(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=100,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="five_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(5, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_3",
            problem_fn=lambda: move_around_blockers_two(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=100,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(6, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=32,
        ),
    ],
)


green_belt_practice = build_lesson_plan(
    "green_belt_practice",
    [
        LessonExercise(
            lesson_name="commute_grouping_1",
            problem_fn=lambda: combine_terms_after_commuting(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            max_turns=4,
            mcts_sims=250,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
            num_observations=128,
        ),
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(6, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="eight_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(8, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=200,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="ten_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=500,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_grouping_3",
            problem_fn=lambda: combine_terms_after_commuting(8, 15, 3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=250,
            num_observations=32,
            max_turns=6,
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

green_belt = build_lesson_plan(
    "green_belt",
    [
        LessonExercise(
            lesson_name="six_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(6, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="eight_terms_with_exponents",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(8, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_blockers_1_7",
            problem_fn=lambda: move_around_blockers_one(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=50,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="ten_terms_with_exponents",
            problem_fn=lambda: simplify_multiple_terms(10, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=50,
            num_observations=32,
        ),
        LessonExercise(
            lesson_name="commute_blockers_2_7",
            problem_fn=lambda: move_around_blockers_two(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            problem_count=4,
            mcts_sims=50,
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
            problem_fn=lambda: combine_terms_in_place(easy=False, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            max_turns=3,
            num_observations=256,
        ),
        LessonExercise(
            lesson_name="three_terms_with_powers",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(3, powers=True),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
            num_observations=256,
        ),
        LessonExercise(
            lesson_name="seven_terms_with_powers",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7, powers=True),
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
