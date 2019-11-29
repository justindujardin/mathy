Mathy parses [token arrays](/cas/tokenizer) into symbolic trees that can be inspected, transformed, and visualized.

## Motivation

While a Token array verifies that text maps to some mathematical expression, its form fails to encode the `Order of Operations` that is required to evaluate the end result.

Said another way, a token array is sufficient to make sure there are not errors in an expression, but a tree encoding the order of operations is required to iterate the expression and calculate its numeric value.

## Examples

To help better understand what the parser does, consider a few examples of expressions and their visualized trees:

| Text                  | Tree                        |
| --------------------- | --------------------------- |
| `4x`                  | `mathy:4x`                  |
| `4x / 2y^7`           | `mathy:4x/2y^7`             |
| `4x + (1/3)y + 7x`    | `mathy:4x+ (1/3)y + 7x`     |
| `4x + 1/3y + 7x`      | `mathy:4x+ 1/3y + 7x`       |
| `(28 + 1j)(17j + 2y)` | `mathy:(28 + 1j)(17j + 2y)` |
