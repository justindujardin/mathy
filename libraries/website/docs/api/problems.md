# mathy.problems
Problem Generation
---

Utility functions for helping generate input problem texts.

## binomial_times_binomial
```python
binomial_times_binomial(*, op='+', min_vars=1, max_vars=2, simple_variables=True, powers_probability=0.33, like_variables_probability=1.0) -> Tuple[str, int]
```

## combine_terms_in_place
```python
combine_terms_in_place(min_terms=16, max_terms=26, easy=True, powers=False)
```
Generate a problem that puts one pair of like terms somewhere inside
an expression of unlike terms. The agent should be challenged to make its first
few moves count when combined with a very small number of maximum moves.
The hope is that by focusing the agent on selecting the right moves inside of a
ridiculously large expression it will learn to select actions to combine like terms
invariant of the sequence length.

### Example

```
  4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j
```

`mathy:4y + 12j + 73q + 19k + 13z + 56l + (24x + 12x) + 43n + 17j`


## commute_haystack
```python
commute_haystack(min_terms=5, max_terms=8, commute_blockers=1, easy=True, powers=False)
```
A problem with a bunch of terms that have no matches, and a single
set of two terms that do match, but are separated by one other term.
The challenge is to commute the terms to each other in one move.

### Example

```
4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j"
                              ^-----------^
```

`mathy:4y + 12j + 73q + 19k + 13z + 24x + 56l + 12x  + 43n + 17j`

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
## move_around_blockers_one
```python
move_around_blockers_one(number_blockers:int, powers_probability:float=0.5)
```
Two like terms separated by (n) blocker terms, e.g. `4x + (y + f) + x`

### Example
`mathy:4x + (y + f) + x`
## move_around_blockers_two
```python
move_around_blockers_two(number_blockers:int, powers_probability:float=0.5)
```
Two like terms with three blockers, e.g. `7a + 4x + (2f + j) + x + 3d`
### Example

`mathy:7a + 4x + (2f + j) + x + 3d`
## simplify_multiple_terms
```python
simplify_multiple_terms(num_terms, optional_var=False, op='+', common_variables=True, inner_terms_scaling=0.3, powers_probability=0.33, optional_var_probability=0.8, noise_probability=0.8, shuffle_probability=0.66, noise_terms=None) -> Tuple[str, int]
```
Generate a polynomial problem with like terms that need to be combined and
simplified, e.g. `2a + 3j - 7b + 17.2a + j`

`mathy:2a + 3j - 7b + 17.2a + j`

## solve_for_variable
```python
solve_for_variable(terms=4)
```
Generate a solve for x type problem, e.g. `4x + 2 = 8x`

`mathy:4x + 2 = 8x`
## split_in_two_random
```python
split_in_two_random(value:int)
```
Split a given number into two smaller numbers that sum to it.
Returns: a tuple of (lower, higher) numbers that sum to the input

