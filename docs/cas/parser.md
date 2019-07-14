## Motivation

While an array of Tokens can assure that a piece of text maps to some complete mathematical expression, its form fails to encode the `Order of Operations` that is required to evaluate the end result. 

Said another way, a token array is sufficient to make sure there are not errors in an expression, but a tree encoding the order of operations is required to iterate the expression and calculate its numeric value.

