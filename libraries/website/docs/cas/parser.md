Mathy parses problem texts into symbolic trees that can be inspected, transformed, and visualized.

??? Nay Sayer

    Why not deal with text or tokens directly? Trees encode the order-of-operations that is required
    for evaluating or manipulating expressions with confidence.

    `mathy:(42 / 13y) -7x`

## Motivation

While a Token array verifies that text maps to some mathematical expression, its form fails to encode the `Order of Operations` that is required to evaluate the end result.

Said another way, a token array is sufficient to make sure there are not errors in an expression, but a tree encoding the order of operations is required to iterate the expression and calculate its numeric value.
