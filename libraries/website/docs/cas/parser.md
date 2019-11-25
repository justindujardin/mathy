## Motivation

While a Token array verifies that text maps to some mathematical expression, its form fails to encode the `Order of Operations` that is required to evaluate the end result. 

Said another way, a token array is sufficient to make sure there are not errors in an expression, but a tree encoding the order of operations is required to iterate the expression and calculate its numeric value.

