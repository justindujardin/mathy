import random
import sys

operators = list("+*")
common_variables = list("xyz")
variables = list("abcdefghijklmnopqrstuvwxyz")
max_const = 24


def rand_bool(percent_chance=None):
    if percent_chance is None:
        percent_chance = 50
    return bool(random.randrange(100) < percent_chance)


def rand_var(common=False):
    if common is True:
        return common_variables[random.randint(0, len(common_variables) - 1)]
    return variables[random.randint(0, len(variables) - 1)]


def maybe_var(percent_chance=80, common_var=False):
    return rand_var(common_var) if rand_bool(percent_chance) else ""


def maybe_int(percent_chance=80):
    return rand_int() if rand_bool(percent_chance) else ""


def maybe_power(percent_chance=80, max_power=4):
    if rand_bool(percent_chance):
        return "^{}".format(random.randint(2, max_power))
    else:
        return ""


def rand_int():
    return random.randint(1, max_const)


def rand_op():
    return operators[random.randint(0, len(operators) - 1)]


def get_rand_vars(num_vars, exclude_vars=[], common_variables=False):
    """Get a list of random variables, excluding the given list of hold-out variables"""
    var = rand_var()
    if num_vars > 25:
        raise ValueError("out of range: there are only twenty-six variables")
    rand_vars = set()
    while len(rand_vars) < num_vars:
        _rand = rand_var(common_variables)
        if _rand not in exclude_vars:
            rand_vars.add(_rand)
    out = list(rand_vars)
    random.shuffle(out)
    return out


def combine_multiple_like_add_terms(num_terms, optional_var=False):
    variable = rand_var()
    # Guarantee at least one set of like terms
    result = "{}{}".format(rand_int(), variable)
    suffix = " + {}{}".format(rand_int(), variable)
    for i in range(num_terms - 2):
        result = result + " + {}{}".format(
            rand_int(), maybe_var() if optional_var else rand_var()
        )
    return result + suffix, num_terms


def simplify_multiple_terms(
    num_terms,
    optional_var=False,
    op="+",
    common_variables=True,
    powers=False,
    inner_terms_scaling=0.3,
):
    # Generate from common varible names to have more chance of
    # sets of like terms.
    variable = rand_var(common_variables)
    # Guarantee at least one set of terms with a common variable. This ensures
    # that the problem has at least one operation that must be done (resolve the conflict
    # between the two matching variable terms.)
    power_percent_chance = 80 if powers == True else 0
    pre_power = maybe_power(power_percent_chance)
    result = "{}{}{}".format(rand_int(), variable, pre_power)
    result = "{}{}{}".format(rand_int(), variable, pre_power)
    suffix = " {} {}{}{}".format(
        rand_op() if op is None else op, rand_int(), variable, pre_power
    )
    var_powers = {}

    # This is made up on the fly. The idea is that you subtract the two (bookend terms)
    num_like_terms = max(1, int((num_terms - 2) * inner_terms_scaling))
    other_vars = get_rand_vars(num_like_terms, exclude_vars=[variable]) * 10
    for i in range(num_terms - 2):
        other_var = other_vars[i]
        if optional_var and rand_bool() is False:
            other_var = ""
        result = result + " {} {}{}".format(
            rand_op() if op is None else op, rand_int(), other_var
        )
    return result + suffix, num_terms


def solve_for_variable(terms=4):
    """Generate a solve for x type problem, e.g. `4x + 2 = 8x`"""
    variable = rand_var()
    # Guarantee at least one set of like terms
    result = "{}{} = {}".format(rand_int(), variable, rand_int())
    suffix = " + {}{}".format(rand_int(), variable)
    for _ in range(terms - 3):
        num = rand_int()
        op = rand_op()
        var = optional_var()
        result = result + " {} {}{}".format(op, num, var)
    return result + suffix


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
