The `Constant Arithmetic` rule transforms an expression tree by combining two constant values that are separated by a binary operation like `addition` or `division`. 

Examples:

 - `4 + 2` becomes `6`
 - `2x * 4` becomes `8x`
 - `4 / 1` becomes `4`
 - `12 - 2.5` becomes `9.5`

!!! note
    The constant simplify rule has the ability to simplify constants across a sibling when the sibling is a variable chunk and the constants are commutatively connected.

    For example `2x * 8` can be transformed into `16x` because the constants are connected to each other through a multiplication chain that allows commuting. 

### Transformations

#### Two Constants

The simplest transform is to evaluate two constants that are siblings.

- `(4 * 2) + 3`

#### Sibling Skipping

When the commutative property is satisfied we can combine two constants separated by a variable sibling.

- `(4n * 2) + 3`

#### Alternate Tree Forms

Math trees can be represented in a number of different equivalent forms, so mathy supports these unnatural groupings to make this rule applicable to more nodes in the tree.

- `5 * (8h * t)`
- `(7 * 10y^3) * x`
- `(7q * 10y^3) * x`
- `792z^4 * 490f * q^3`
- `(u^3 * 36c^6) * 7u^3`
