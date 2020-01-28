# mathy.util

## calculate_grouping_control_signal
```python
calculate_grouping_control_signal(
    input: str,
    output: str,
    clip_at_zero: bool = False,
) -> float
```
Calculate grouping_control signals as the sum of all distances between
all like terms. Gather all the terms in an expression and add an error value
whenever a like term is separated by another term.
## compare_expression_string_values
```python
compare_expression_string_values(
    from_expression: str,
    to_expression: str,
    history: Union[List[Any], NoneType] = None,
)
```
Compare and evaluate two expressions strings to verify they have the
same value
## compare_expression_values
```python
compare_expression_values(
    from_expression: mathy.core.expressions.MathExpression,
    to_expression: mathy.core.expressions.MathExpression,
    history: Union[List[Any], NoneType] = None,
)
```
Compare and evaluate two expressions to verify they have the same value
## discount
```python
discount(values:List[float], gamma=0.99) -> List[float]
```
Discount a list of floating point values.

__Arguments__

- __r (List[float])__: the list of floating point values to discount
- __gamma (float)__: a value between 0 and 0.99 to use when discounting the inputs

__Returns__

`(List[float])`: a list of the same size as the input with discounted values

## get_term_ex
```python
get_term_ex(
    node: Union[mathy.core.expressions.MathExpression, NoneType],
) -> Union[mathy.util.TermEx, NoneType]
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

## get_terms
```python
get_terms(
    expression: mathy.core.expressions.MathExpression,
) -> List[mathy.core.expressions.MathExpression]
```
Walk the given expression tree and return a list of nodes
representing the distinct terms it contains.

__Arguments__

- __expression (MathExpression)__: the expression to find term nodes in

__Returns__

`(List[MathExpression])`: a list of term nodes

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

Return True if a given term has been simplified so that it only has
a max of one coefficient and variable, with the variable on the right
and the coefficient on the left side

Examples

  - Complex   = 2 * 2x^2
  - Simple    = x^2 * 4
  - Preferred = 4x^2

## is_simple_term
```python
is_simple_term(node:mathy.core.expressions.MathExpression) -> bool
```
Return True if a given term has been simplified so it only has at
most one of each variable and a constant.

__Examples__

  - Simple = 2x^2 * 2y
  - Complex = 2x * 2x * 2y
  - Simple = x^2 * 4
  - Complex = 2 * 2x^2

## pad_array
```python
pad_array(in_list:List[Any], max_length:int, value:Any=0) -> List[Any]
```
Pad a list to the given size with the given padding value.

__Arguments:__

in_list (List[Any]): List of values to pad to the given length
max_length (int): The desired length of the array
value (Any): a value to insert in order to pad the array to max length

__Returns__

`(List[Any])`: An array padded to `max_length` size

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
terms_are_like(
    one: Union[mathy.util.TermResult, mathy.core.expressions.MathExpression, typing_extensions.Literal[False]],
    two: Union[mathy.util.TermResult, mathy.core.expressions.MathExpression, typing_extensions.Literal[False]],
) -> bool
```
Determine if two math expression nodes are **like terms**.

__Arguments__

- __one (MathExpression)__: A math expression that represents a term
- __two (MathExpression)__: Another math expression that represents a term

__Returns__

`(bool)`: Whether the terms are like or not.

## unlink
```python
unlink(
    node: Union[mathy.core.expressions.MathExpression, NoneType] = None,
) -> Union[mathy.core.expressions.MathExpression, NoneType]
```
Unlink an expression from it's parent.

1. Clear expression references in `parent`
2. Clear `parent` in expression

