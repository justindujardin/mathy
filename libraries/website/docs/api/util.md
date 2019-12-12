# mathy.util

## calculate_grouping_control_signal
```python
calculate_grouping_control_signal(input:str, output:str, clip_at_zero:bool=False) -> float
```
Calculate grouping_control signals as the sum of all distances between
all like terms. Gather all the terms in an expression and add an error value
whenever a like term is separated by another term.

Examples:
    "2x + 2x" = 0
    "2x + 4y + 2x" = 1
    "2x + 4y + 2x + 4y" = 2
    "2x + 2x + 4y + 4y" = 0

## compare_expression_string_values
```python
compare_expression_string_values(from_expression:str, to_expression:str, history:Union[List[Any], NoneType]=None)
```
Compare and evaluate two expressions strings to verify they have the
same value
## compare_expression_values
```python
compare_expression_values(from_expression:mathy.core.expressions.MathExpression, to_expression:mathy.core.expressions.MathExpression, history:Union[List[Any], NoneType]=None)
```
Compare and evaluate two expressions to verify they have the same value
## discount
```python
discount(r, gamma=0.99)
```
Discount a list of float rewards to encourage rapid convergance.
r: input array of floats
gamma: a float value between 0 and 0.99
## get_term_ex
```python
get_term_ex(node:Union[mathy.core.expressions.MathExpression, NoneType]) -> Union[mathy.util.TermEx, NoneType]
```
Extract the 3 components of a naturally ordered term.

!!! info Important

    This doesn't care about whether the node is part of a larger term,
    it only looks at its children.

__Example__


`mathy:4x^7`

```python
TermEx(coefficient=4, variable="x", exponent=7)
```

## has_like_terms
```python
has_like_terms(expression:mathy.core.expressions.MathExpression) -> bool
```
Return True if a given expression has more than one of any type of term.

__Examples__


- `x + y + z` = `False`
- `x^2 + x` = `False`
- `y + 2x` = `True`
- `x^2 + 4x^3 + 2y` = `True`

## is_debug_mode
```python
is_debug_mode()
```
Debug mode enables extra logging and assertions, but is slower because of
the increased sanity check measurements.
## is_preferred_term_form
```python
is_preferred_term_form(expression:mathy.core.expressions.MathExpression) -> bool
```

Return True if a given term has been simplified such that it only has
a max of one coefficient and variable, with the variable on the right
and the coefficient on the left side

__Example__


    Complex   = 2 * 2x^2
    Simple    = x^2 * 4
    Preferred = 4x^2

## is_simple_term
```python
is_simple_term(node:mathy.core.expressions.MathExpression) -> bool
```

Return True if a given term has been simplified so it only has at
most one of each variable and a constant.

__Examples__


    Simple = 2x^2 * 2y
    Complex = 2x * 2x * 2y

    Simple = x^2 * 4
    Complex = 2 * 2x^2

## negate
```python
negate(node)
```
Negate an addition node: `l - r = l + (-r)`

```
            -                  +
           / \                / \
          /   \     ->       /   \
         /     \            /     \
        *       2          *       -
       / \                / \       \
      4   x              4   x       2
```

## normalize_rewards
```python
normalize_rewards(r)
```
Normalize a set of rewards to values between -1 and 1
## pad_array
```python
pad_array(A, max_length, value=0, backwards=False, cleanup=False)
```
Pad a list to the given size with the given padding value

If backwards=True the input will be reversed after padding, and
the output will be reversed after padding, to correctly pad for
LSTMs, e.g. "4x+2----" padded backwards would be "----2+x4"

## TermEx
```python
TermEx(self, /, *args, **kwargs)
```
TermEx(coefficient, variable, exponent)
### coefficient
An optional integer or float coefficient
### exponent
An optional integer or float exponent
### variable
An optional variable
## terms_are_like
```python
terms_are_like(one, two)
```

@param {Object|MathExpression} one The first term {@link `get_term`}
@param {Object|MathExpression} two The second term {@link `get_term`}
@returns {Boolean} Whether the terms are like or not.

## unlink
```python
unlink(node:Union[mathy.core.expressions.MathExpression, NoneType]=None) -> Union[mathy.core.expressions.MathExpression, NoneType]
```
Unlink an expression from it's parent.

1. Clear expression references in `parent`
2. Clear `parent` in expression

