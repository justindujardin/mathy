# mathy.envs.complex_simplify

## ComplexSimplify
```python
ComplexSimplify(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for simplifying complex terms (e.g. 4x^3 * 7y) inside of
expressions. The goal is to simplify the complex term within the allowed number
of environment steps.

### problem_fn
```python
ComplexSimplify.problem_fn(self, params:mathy.types.MathyEnvProblemArgs) -> mathy.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
a complex term that has a simple representation that must be found.
- "4x * 2y^2 * 7q"
- "7j * 2z^6"
- "x * 2y^7 * 8z * 2x"

