Mathy parses [token arrays](/cas/tokenizer) into inspectable, transformable, visualizable symbolic trees.

## Motivation

A Token array verifies that text maps to some known set of symbols, but not that they are a correct ordering that produces a valid mathematical expression. The mathy Parser class converts tokens into a tree while also validating that the tree follows the expected Order of Operations.

## Examples

To help better understand what the parser does, consider a few examples of expressions and their visualized trees:

| Text                  | Tree                        |
| --------------------- | --------------------------- |
| `4x`                  | `mathy:4x`                  |
| `4x / 2y^7`           | `mathy:4x/2y^7`             |
| `4x + (1/3)y + 7x`    | `mathy:4x+ (1/3)y + 7x`     |
| `4x + 1/3y + 7x`      | `mathy:4x+ 1/3y + 7x`       |
| `(28 + 1j)(17j + 2y)` | `mathy:(28 + 1j)(17j + 2y)` |
