# mathy.problems
Problem Generation
---

Utility functions for helping generate input problems.

## gen_binomial_times_binomial
```python
gen_binomial_times_binomial(
    op = '+',
    min_vars = 1,
    max_vars = 2,
    simple_variables = True,
    powers_probability = 0.33,
    like_variables_probability = 1.0,
) -> Tuple[str, int]
```
Generate a binomial multiplied by another binomial.

__Example__


```
(2e + 12p)(16 + 7e)
```

`mathy:(2e + 12p)(16 + 7e)`

## gen_binomial_times_monomial
```python
gen_binomial_times_monomial(
    op = '+',
    min_vars = 1,
    max_vars = 2,
    simple_variables = True,
    powers_probability = 0.33,
    like_variables_probability = 1.0,
) -> Tuple[str, int]
```
Generate a binomial multiplied by a monomial.

__Example__


```
(4x^3 + y) * 2x
```

`mathy:(4x^3 + y) * 2x`

## gen_combine_terms_in_place
```python
gen_combine_terms_in_place(
    min_terms = 16,
    max_terms = 26,
    easy = True,
    powers = False,
) -> Tuple[str, int]
```
Generate a problem that puts one pair of like terms next to each other
somewhere inside a large tree of unlike terms.

The problem is intended to be solved in a very small number of moves, making
training across many episodes relatively quick, and reducing the combinatorial
explosion of branches that need to be searched to solve the task.

The hope is that by focusing the agent on selecting the right moves inside of a
ridiculously large expression it will learn to select actions to combine like terms
invariant of the sequence length.

__Example__


```
4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j
```

`mathy:4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j`


## gen_commute_haystack
```python
gen_commute_haystack(
    min_terms = 5,
    max_terms = 8,
    commute_blockers = 1,
    easy = True,
    powers = False,
)
```
A problem with a bunch of terms that have no matches, and a single
set of two terms that do match, but are separated by one other term.
The challenge is to commute the terms to each other in one move.

__Example__


```
4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j"
                              ^-----------^
```

`mathy:4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j`

## gen_move_around_blockers_one
```python
gen_move_around_blockers_one(number_blockers:int, powers_probability:float=0.5)
```
Two like terms separated by (n) blocker terms.

__Example__


```
4x + (y + f) + x
```

`mathy:4x + (y + f) + x`
## gen_move_around_blockers_two
```python
gen_move_around_blockers_two(number_blockers:int, powers_probability:float=0.5)
```
Two like terms with three blockers.

__Example__


```
7a + 4x + (2f + j) + x + 3d
```

`mathy:7a + 4x + (2f + j) + x + 3d`
## gen_move_around_interleaved_like_terms
```python
gen_move_around_interleaved_like_terms(number_terms, number_pairs)
```
Interleaved multiple like variables.

__Example__


```
4x + 2y + 6x + 3y
```

`mathy:4x + 2y + 6x + 3y`

## gen_simplify_multiple_terms
```python
gen_simplify_multiple_terms(
    num_terms: int,
    optional_var: bool = False,
    op: Union[List[str], str] = '+',
    common_variables: bool = True,
    inner_terms_scaling: float = 0.3,
    powers_probability: float = 0.33,
    optional_var_probability: float = 0.8,
    noise_probability: float = 0.8,
    shuffle_probability: float = 0.66,
    noise_terms: int = None,
) -> Tuple[str, int]
```
Generate a polynomial problem with like terms that need to be combined and
simplified.

__Example__


```
2a + 3j - 7b + 17.2a + j
```

`mathy:2a + 3j - 7b + 17.2a + j`

## gen_solve_for_variable
```python
gen_solve_for_variable(terms=4) -> Tuple[str, int]
```
Generate a solve for x type problem.

__Example__


```
4x + 2 = 8x
```

`mathy:4x + 2 = 8x`

## get_blocker
```python
get_blocker(num_blockers=1, exclude_vars=[])
```
Get a string of terms to place between target simplification terms
in order to challenge the agent's ability to use commutative/associative
rules to move terms around.
## get_rand_vars
```python
get_rand_vars(num_vars, exclude_vars=[], common_variables=False)
```
Get a list of random variables, excluding the given list of hold-out variables
## split_in_two_random
```python
split_in_two_random(value:int)
```
Split a given number into two smaller numbers that sum to it.
Returns: a tuple of (lower, higher) numbers that sum to the input

