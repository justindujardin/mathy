# Associative Property

The `Associative Property` of numbers says that we can re-group two `addition` or `multiplication` terms such that one is evaluated before the other, without changing the value of the expression.

The formulation of this property is the same for addition and multiplication:

- Addition `(a + b) + c = a + (b + c)`
- Multiplication `(a * b) * c = a * (b * c)`

Examples:

- `(4 + 3) + 2 = 9` and `4 + (3 + 2) = 9`
- `2 * (3 * 1) = 6` and `(2 * 3) * 1 = 6`

!!! note
    Interestingly, the application of the associative property of numbers to a binary expression tree is a common tree operation called a "node rotation."


### Transformations

#### Addition

```
(a + b) + c = a + (b + c)

     (y) +            + (x)
        / \          / \
       /   \        /   \
  (x) +     c  ->  a     + (y)
     / \                / \
    /   \              /   \
   a     b            b     c
```

#### Multiplication

```
(a * b) * c = a * (b * c)

     (x) *            * (y)
        / \          / \
       /   \        /   \
  (y) *     c  <-  a     * (x)
     / \                / \
    /   \              /   \
   a     b            b     c
```
