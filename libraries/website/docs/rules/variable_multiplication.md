The `Variable Multiplication` rule restates `x^b * x^d` as `x^(b + d)` which has the effect of isolating the exponents attached to the variables, so they can be combined.

!!! note

        This rule can only be applied when the nodes have matching variable bases. This means that `x * y` cannot be combined, but `x * x` can be.

### Transformations

Both implicit and explicit variable powers are recognized in this transformation.

!!! info "Help Wanted"

        The current variable multiply rule leaves out a case where there is a power
        raised to another power, they can be combined by multiplying the exponents
        together.

        For example: `x^(2^2) = x^4`

#### Explicit powers

In the simplest case both variables have explicit exponents.

Examples: `x^b * x^d = x^(b+d)`

- `42x^2 * x^3` becomes `42x^(2 + 3)`
- `x^1 * x^7` becomes `x^(1 + 8)`

```
            *
           / \
          /   \          ^
         /     \    =   / \
        ^       ^      x   +
       / \     / \        / \
      x   b   x   d      b   d
```

#### Implicit powers

When not explicitly stated, a variable has an implicit power of being raised to the 1, and this form is identified.

Examples: `x * x^d = x^(1 + d)`

- `42x * x^3` becomes `42x^(1 + 3)`
- `x * x` becomes `x^(1 + 1)`

```
            *
           / \
          /   \          ^
         /     \    =   / \
        x       ^      x   +
               / \        / \
              x   d      1   d
```

### Examples

`rule_tests:variable_multiply`
