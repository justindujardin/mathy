A set of binomials are multiplied together and must to be simplified to satisfy the win-conditions.

## Challenge

In Binomial Multiply the agent must learn to quickly distribute the binomial multiplications and factor out common terms to leave a simplified representation.

Examples

- `xxxxxx` must be simplified to `xxxx`

## Win Conditions

A problem is considered solved when there are no remaining complex terms in the expression.

### No Complex Terms

Terms are considered complex when there's a more concise way they could be expressed.

Examples

- `2 * 4x` is **complex** because it has **multiple coefficients** which could be simplified to `8x`
- `4x * y * j^2` is **not complex** despite being verbose because there is only a **single coefficient** and **no matching variables**

## Example Episode

A trained agent learns to distribute and simplify binomial and monomial multiplications.

### Input

`(k^4 + 7)(4 + h^2)`

`mathy:(k^4 + 7)(4 + h^2)`

### Steps

| Step                  | Text                                |
| --------------------- | ----------------------------------- |
| initial               | (k^4 + 7)(4 + h^2)                  |
| distributive multiply | (4 + h^2) \* k^4 + (4 + h^2) \* 7   |
| distributive multiply | 4k^4 + k^4 \* h^2 + (4 + h^2) \* 7  |
| commutative swap      | 4k^4 + k^4 \* h^2 + 7 \* (4 + h^2)  |
| distributive multiply | 4k^4 + k^4 \* h^2 + (7 \* 4 + 7h^2) |
| constant arithmetic   | 4k^4 + k^4 \* h^2 + (28 + 7h^2)     |
| solution              | **4k^4 + k^4 \* h^2 + 28 + 7h^2**   |

### Solution

`4k^4 + k^4 * h^2 + 28 + 7h^2`

`mathy:4k^4 + k^4 * h^2 + 28 + 7h^2`
