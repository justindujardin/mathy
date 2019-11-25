Core to working with algebra problems is the ability to `simplify complex terms`, and Mathy provides an environment that generates problems with complex terms that require simplification to satisfy the win-conditions.

## Challenge

In Complex Simplify the agent must learn to quickly combine coefficients and like variables inside a single complex term

Examples

- `1x * 2x^1` must be simplified to `2x^2`
- `7j * y^1 * y^2` must be simplified to `7j * y^3`

## Win Conditions

A problem is considered solved when there are no remaining complex terms in the expression.

### No Complex Terms

Terms are considered complex when there's a more concise way they could be expressed.

Examples

- `2 * 4x` is **complex** because it has **multiple coefficients** which could be simplified to `8x`
- `4x * y * j^2` is **not complex** despite being verbose because there is only a **single coefficient** and **no matching variables**

## Example Episode

A trained agent learns to combine multiple low-level actions into higher-level ones that `simplify complex terms`

**Input**: `4a^4 * 5a^4 * 2b^4`

| Step                    | Text                        |
| ----------------------- | --------------------------- |
| initial                 | 4a^4 _ 5a^4 _ 2b^4          |
| constant arithmetic     | **20a^4** _ a^4 _ 2b^4      |
| variable multiplication | **20 \* a^(4 + 4)** \* 2b^4 |
| constant arithmetic     | 20 _ **a^8** _ 2b^4         |
| commutative swap        | **(a^8 \* 2b^4)** \* 20     |
| commutative swap        | (**2b^4 \* a^8**) \* 20     |
| commutative swap        | 20 _ \_\_2b^4 _ a^8\_\_     |
| constant arithmetic     | **40b^4** \* a^8            |
| solution                | **40b^4 \* a^8**            |

**Input**:`1k + 210r + 7z + 11k + 10z`

| Step                   | Text                                     |
| ---------------------- | ---------------------------------------- |
| input                  | 1k + 210r + 7z + 11k + 10z               |
| commutative swap       | 11k + **(1k + 210r + 7z)** + 10z         |
| distributive factoring | 11k + (1k + 210r) + **(7 + 10) \* z**    |
| distributive factoring | **(11 + 1) \* k** + 210r + (7 + 10) \* z |
| constant arithmetic    | (11 + 1) \* k + 210r + **17z**           |
| constant arithmetic    | **12k** + 210r + 17z                     |
| solution               | **12k + 210r + 17z**                     |
