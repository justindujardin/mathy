"""Problem Generation
---

Utility functions for helping generate input problems.
"""
import random
from typing import Tuple, Optional, Union, List
from pydantic import BaseModel, Field

operators = list("+*")
common_variables = list("xyz")
variables = list("abcdfghjklmnopqrstuvwxyz")
max_const = 12
_pretty_numbers = True


class MathyTermTemplate(BaseModel):
    variable: Optional[str] = Field(None, description="the term variable")
    exponent: Optional[int] = Field(None, description="the term exponent")

    def make(self) -> str:
        return mathy_term_string(
            coefficient=maybe_number(), exponent=self.exponent, variable=self.variable
        )


class MathyProblemTerm(MathyTermTemplate):
    coefficient: Optional[Union[float, int]] = None


MathyExcludedTermTemplates = Optional[List[MathyTermTemplate]]


def mathy_term_string(
    *,
    coefficient: Optional[Union[int, float]] = None,
    exponent: Optional[Union[int, float]] = None,
    variable: Optional[str] = None,
) -> str:
    pieces = []
    if coefficient is not None:
        pieces.append(f"{coefficient}")
    if variable is not None:
        pieces.append(f"{variable}")
    if exponent is not None:
        pieces.append(f"^{exponent}")
    return "".join(pieces)


def get_rand_term_templates(
    num_templates: int,
    exclude_like: MathyExcludedTermTemplates = None,
    common_variables=False,
    exponent_probability=0.5,
) -> List[MathyTermTemplate]:
    result: List[MathyTermTemplate] = []

    exclude: List[str] = []
    if exclude_like is not None:
        exclude = [
            mathy_term_string(variable=t.variable, exponent=t.exponent)
            for t in exclude_like
        ]

    failures = 0
    while len(result) < num_templates:
        if failures > 100:
            raise EnvironmentError(
                f"failed to generate a random term after {failures} tries!"
            )
        variable = rand_var(common_variables)
        exponent = maybe_number(exponent_probability * 100, None)
        # Don't generate x^1
        if exponent == 1:
            exponent = 2
        key = mathy_term_string(variable=variable, exponent=exponent)
        if key not in exclude:
            result.append(MathyTermTemplate(variable=variable, exponent=exponent))
            exclude.append(key)
        else:
            failures += 1
    random.shuffle(result)
    return result


def rand_bool(percent_chance=None):
    if percent_chance is None:
        percent_chance = 50
    return bool(random.randrange(100) < percent_chance)


def rand_var(common=False):
    if common is True:
        return common_variables[random.randint(0, len(common_variables) - 1)]
    return variables[random.randint(0, len(variables) - 1)]


def maybe_var(percent_chance=80, common_var=False, or_else=""):
    return rand_var(common_var) if rand_bool(percent_chance) else or_else


def maybe_number(percent_chance=80, or_else=""):
    return rand_number() if rand_bool(percent_chance) else or_else


def maybe_power(percent_chance=80, max_power=4, or_else=""):
    if rand_bool(percent_chance):
        return "^{}".format(random.randint(2, max_power))
    else:
        return or_else


def use_pretty_numbers(enabled: bool = True):
    """Determine if problems should include only pretty numbers or 
    a whole range of integers and floats. Using pretty numbers will
    restrict the numbers that are generated to integers between 1 and
    12. When not using pretty numbers, floats and large integers will
    be included in the output from `rand_number`"""
    global _pretty_numbers
    _pretty_numbers = enabled


def rand_number():
    global _pretty_numbers
    if _pretty_numbers:
        min_value = 1
        max_value = 12
    else:
        min_value = -10000
        max_value = 10000
    # Use an integer?
    if _pretty_numbers or rand_bool(66):
        if _pretty_numbers or rand_bool(50):
            result = random.randint(min_value, max_value)
        else:
            result = random.randint(1, max_const)
    else:
        if rand_bool(10):
            result = random.random() * max_value
        elif rand_bool(10):
            result = random.random() * min_value
        else:
            result = random.random() * max_const

        result = truncate(result, max_decimals=1)
    return result


def truncate(value, max_decimals=3):
    return float(f"%.{max_decimals}f" % (float(value)))


def rand_op():
    return operators[random.randint(0, len(operators) - 1)]


def get_rand_vars(num_vars, exclude_vars=[], common_variables=False):
    """Get a list of random variables, excluding the given list of hold-out variables"""
    if num_vars > 25:
        raise ValueError("out of range: there are only twenty-six variables")
    rand_vars = set()
    iters = 0
    while len(rand_vars) < num_vars:
        _rand = rand_var(common_variables)
        if _rand not in exclude_vars:
            rand_vars.add(_rand)
        iters += 1
        if iters > num_vars * 10:
            raise ValueError(
                f"Unable to fulfill request for {num_vars} random variables"
            )
    out = list(rand_vars)
    random.shuffle(out)
    return out


def gen_binomial_times_binomial(
    *,
    op="+",
    min_vars=1,
    max_vars=2,
    simple_variables=True,
    powers_probability=0.33,
    like_variables_probability=1.0,
) -> Tuple[str, int]:
    """Generate a binomial multiplied by another binomial.

    # Example

    ```
    (2e + 12p)(16 + 7e)
    ```

    `mathy:(2e + 12p)(16 + 7e)`
    """
    power_prob_percent = powers_probability * 100
    powers = rand_bool(power_prob_percent)
    like_vars = rand_bool(like_variables_probability * 100)

    num_terms: int = 4

    num_vars: int = random.randint(min_vars, max_vars)
    terms = [""] * 4

    # Build variables (with optional exponents)
    if like_vars is False:
        for i, var in enumerate(get_rand_vars(num_vars)):
            if powers is not False:
                terms[i] = f"{var}{maybe_power(power_prob_percent * 2)}"
            else:
                terms[i] = var
    else:
        var = rand_var()
        if powers is not False:
            var = f"{var}{maybe_power(power_prob_percent * 2)}"
        for i in range(num_vars):
            terms[i] = var
    # random.shuffle(terms)

    # Conditionally attach coefficients to each term
    for i in range(4):
        if simple_variables is True and terms[i] != "":
            continue
        terms[i] = f"{rand_number()}{terms[i]}"

    first = [terms[0], terms[2]]
    second = [terms[1], terms[3]]
    random.shuffle(first)
    random.shuffle(second)
    return f"({first[0]} + {first[1]})({second[0]} + {second[1]})", num_terms + 2


def gen_binomial_times_monomial(
    *,
    op="+",
    min_vars=1,
    max_vars=2,
    simple_variables=True,
    powers_probability=0.33,
    like_variables_probability=1.0,
) -> Tuple[str, int]:
    """Generate a binomial multiplied by a monomial.

    # Example

    ```
    (4x^3 + y) * 2x
    ```

    `mathy:(4x^3 + y) * 2x`
    """
    power_prob_percent = powers_probability * 100
    powers = rand_bool(power_prob_percent)
    like_vars = rand_bool(like_variables_probability * 100)

    num_terms: int = 3

    num_vars: int = random.randint(min_vars, max_vars)
    terms = [""] * num_terms

    # Build variables (with optional exponents)
    if like_vars is False:
        for i, var in enumerate(get_rand_vars(num_vars)):
            if powers is not False:
                terms[i] = f"{var}{maybe_power(power_prob_percent * 2)}"
            else:
                terms[i] = var
    else:
        var = rand_var()
        if powers is not False:
            var = f"{var}{maybe_power(power_prob_percent * 2)}"
        for i in range(num_vars):
            terms[i] = var
    # random.shuffle(terms)

    # Conditionally attach coefficients to each term
    for i in range(num_terms):
        if simple_variables is True and terms[i] != "":
            continue
        terms[i] = f"{rand_number()}{terms[i]}"

    first = [terms[0], terms[2]]
    second = terms[1]
    random.shuffle(first)
    return f"({first[0]} + {first[1]}) * {second}", num_terms


def gen_simplify_multiple_terms(
    num_terms: int,
    optional_var: bool = False,
    op: Union[List[str], str] = "+",
    common_variables: bool = True,
    inner_terms_scaling: float = 0.3,
    powers_probability: float = 0.33,
    optional_var_probability: float = 0.8,
    noise_probability: float = 0.8,
    shuffle_probability: float = 0.66,
    noise_terms: int = None,
) -> Tuple[str, int]:
    """Generate a polynomial problem with like terms that need to be combined and
    simplified.

    # Example

    ```
    2a + 3j - 7b + 17.2a + j
    ```

    `mathy:2a + 3j - 7b + 17.2a + j`
    """
    power_prob_percent = powers_probability * 100
    num_like_terms = max(2, int(num_terms * inner_terms_scaling))
    if num_terms <= 1:
        raise ValueError("num_terms must be at least 2")
    if num_terms == 2:
        num_like_terms = 1
    like_term_vars = get_rand_vars(num_like_terms)
    term_templates = like_term_vars[:]
    for i, var in enumerate(term_templates):
        term_templates[i] = f"{var}{maybe_power(power_prob_percent)}"

    complexity = num_terms

    # Repeat enough times to satisfy max_terms
    term_templates *= num_terms
    term_templates = term_templates[0:num_terms]

    # sometimes add noise terms to the ends
    if rand_bool(noise_probability * 100) is True:
        num_noise_terms = min(5, max(1, num_terms // 3))
        if noise_terms is not None:
            num_noise_terms = noise_terms
        noise_vars = get_rand_vars(num_noise_terms, like_term_vars)

        # When there's noise add complexity
        complexity += 1

        # We take the larger value for the left side to push the terms
        # that have to be matched to the right side of expression. This is
        # so that the model cannot use its existing knowledge about distributive
        # factoring on smaller problems to solve this problem.
        right_num, left_num = split_in_two_random(num_noise_terms)
        for i in range(left_num):
            current = noise_vars.pop()
            term_templates.insert(0, f"{current}{maybe_power(power_prob_percent)}")
        for i in range(right_num):
            current = noise_vars.pop()
            term_templates.append(f"{current}{maybe_power(power_prob_percent)}")

    # sometimes shuffle the terms
    if rand_bool(shuffle_probability * 100) is True:
        random.shuffle(term_templates)

    def get_op() -> str:
        if op is None:
            return rand_op()
        if isinstance(op, list):
            return random.choice(op)
        return op

    root_term = term_templates.pop(0)
    result = f"{rand_number()}{root_term}"
    for other_var in term_templates:
        # other_var = term_templates[i]
        if optional_var and rand_bool(optional_var_probability * 100) is False:
            other_var = ""
        result += f" {get_op()} {rand_number()}{other_var}"
    return result, complexity


def split_in_two_random(value: int):
    """Split a given number into two smaller numbers that sum to it.
    Returns: a tuple of (lower, higher) numbers that sum to the input
    """
    factor = random.uniform(0, 1)
    left = int(factor * value)
    right = value - left
    # always return lower/higher
    return min(left, right), max(left, right)


def gen_combine_terms_in_place(
    min_terms=16, max_terms=26, easy=True, powers=False
) -> Tuple[str, int]:
    """Generate a problem that puts one pair of like terms next to each other
    somewhere inside a large tree of unlike terms.

    The problem is intended to be solved in a very small number of moves, making
    training across many episodes relatively quick, and reducing the combinatorial
    explosion of branches that need to be searched to solve the task.

    The hope is that by focusing the agent on selecting the right moves inside of a
    ridiculously large expression it will learn to select actions to combine like terms
    invariant of the sequence length.

    # Example

    ```
    4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j
    ```

    `mathy:4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j`

    """

    total_terms = random.randint(min_terms, max_terms)
    var = rand_var()
    power_chance = 80 if powers is True else 0
    power = maybe_power(power_chance)
    focus_chunk = f"{maybe_number()}{var}{power} + {maybe_number()}{var}{power}"
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
        out_terms.append(f"{maybe_number()}{current}{maybe_power(power_chance)}")
    out_terms.append(focus_chunk)
    for i in range(right_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_number()}{current}{maybe_power(power_chance)}")

    complexity = total_terms
    problem = " + ".join(out_terms)
    return problem, complexity


def gen_commute_haystack(
    min_terms=5, max_terms=8, commute_blockers=1, easy=True, powers=False
):
    """A problem with a bunch of terms that have no matches, and a single
    set of two terms that do match, but are separated by one other term.
    The challenge is to commute the terms to each other in one move.

    # Example

    ```
    4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j"
                                  ^-----------^
    ```

    `mathy:4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j`
    """
    total_terms = random.randint(min_terms, max_terms)
    num_noise_terms = max(total_terms - 2, commute_blockers)
    var = rand_var()
    noise_vars = get_rand_vars(num_noise_terms, [var])
    power_chance = 80 if powers is True else 0
    power = maybe_power(power_chance)

    # Build up the blockers to put between the like terms
    blockers = []
    for i in range(commute_blockers):
        current = noise_vars.pop()
        blockers.append(f"{maybe_number()}{current}{maybe_power(power_chance)}")

    blocks = " + ".join(blockers)
    focus_chunk = (
        f"{maybe_number()}{var}{power} + {blocks} + {maybe_number()}{var}{power}"
    )
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
        out_terms.append(f"{maybe_number()}{current}{maybe_power(power_chance)}")
    out_terms.append(focus_chunk)
    for i in range(right_num):
        current = noise_vars.pop()
        out_terms.append(f"{maybe_number()}{current}{maybe_power(power_chance)}")

    complexity = len(out_terms)
    problem = " + ".join(out_terms)
    return problem, complexity


def get_blocker(num_blockers=1, exclude_vars=[]):
    """Get a string of terms to place between target simplification terms
    in order to challenge the agent's ability to use commutative/associative
    rules to move terms around."""
    vars = get_rand_vars(num_blockers, exclude_vars)
    out_terms = []
    for i in range(num_blockers):
        out_terms.append("{}{}".format(maybe_number(), vars[i]))
    return " + ".join(out_terms)


def gen_move_around_blockers_one(number_blockers: int, powers_probability: float = 0.5):
    """Two like terms separated by (n) blocker terms.

    # Example

    ```
    4x + (y + f) + x
    ```

    `mathy:4x + (y + f) + x`"""
    var = rand_var()
    power_chance = powers_probability * 100
    exp = maybe_power(power_chance)
    complexity = 2 + number_blockers
    blockers = get_blocker(number_blockers, [var])
    problem = "{}{}{} + {} + {}{}{}".format(
        maybe_number(), var, exp, blockers, maybe_number(), var, exp
    )
    return problem, complexity


def gen_move_around_blockers_two(number_blockers: int, powers_probability: float = 0.5):
    """Two like terms with three blockers.

    # Example

    ```
    7a + 4x + (2f + j) + x + 3d
    ```

    `mathy:7a + 4x + (2f + j) + x + 3d`"""
    rand_vars = get_rand_vars(3)
    [one_var, two_var, three_var] = rand_vars
    complexity = 4 + number_blockers
    power_chance = powers_probability * 100
    one_exp = maybe_power(power_chance)
    two_exp = maybe_power(power_chance)
    three_exp = maybe_power(power_chance)
    problem = "{}{}{} + {}{}{} + {} + {}{}{} + {}{}{}".format(
        maybe_number(),
        one_var,
        one_exp,
        maybe_number(),
        two_var,
        two_exp,
        get_blocker(number_blockers, rand_vars),
        maybe_number(),
        two_var,
        two_exp,
        maybe_number(),
        three_var,
        three_exp,
    )
    return problem, complexity
