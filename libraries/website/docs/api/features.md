# mathy.features

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

