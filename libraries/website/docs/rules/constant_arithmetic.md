The `Constant Arithmetic` rule transforms an expression tree by combining two constant values that are separated by a binary operation like `addition` or `division`.

### Transformations

#### Two Constants

The simplest transform is to evaluate two constants that are siblings.

- `(4 * 2) + 3` = `8 + 3`

#### Sibling Skipping

The constant simplify rule has the ability to simplify constants across a sibling when the sibling is a variable chunk and the constants are commutatively connected.

For example `2x * 8` can be transformed into `16x` because the constants are connected to each other through a multiplication chain that allows [commuting](/rules/commutative_property).

We can see this by taking a look at the trees for `2x * 8` and `2 * 8 * x` and recalling that the commutative proeprty says `a * b = b * a`:

| Satifying the Commutative Property |                   |
| :--------------------------------: | :---------------- |
|           `mathy:2x * 8`           | `mathy:2 * 8 * x` |

We can see that the tree structure has been flipped, but that multiplication nodes still connect the same variables and constants, so the value of the expression remains unchanged.

#### Alternate Tree Forms

Math trees can be represented in a number of different equivalent forms, so mathy supports these unnatural groupings to make this rule applicable to more nodes in the tree.

- `5 * (8h * t)` = `40h * t`
- `(7 * 10y^3) * x` = `70y^3 * x`
- `(7q * 10y^3) * x` = `(70q * y^3) * x`
- `792z^4 * 490f * q^3` = `388080z^4 * f * q^3`
- `(u^3 * 36c^6) * 7u^3` = `u^3 * 252c^6 * u^3`

### Examples

`rule_tests:constants_simplify`
