# mathy.envs.poly_simplify

## PolySimplify
```python
PolySimplify(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for simplifying polynomial expressions.

NOTE: This environment only generates polynomial problems with
 addition operations. Subtraction, Multiplication and Division
 operators are excluded. This is a good area for improvement.

### transition_fn
```python
PolySimplify.transition_fn(self, env_state:mathy.state.MathyEnvState, expression:mathy.core.expressions.MathExpression, features:mathy.state.MathyObservation) -> Union[tf_agents.trajectories.time_step.TimeStep, NoneType]
```
If there are no like terms.
### problem_fn
```python
PolySimplify.problem_fn(self, params:mathy.types.MathyEnvProblemArgs) -> mathy.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
a polynomial problem with (n) total terms divided among (m) groups
of like terms. A few examples of the form: `f(n, m) = p`
- (3, 1) = "4x + 2x + 6x"
- (6, 4) = "4x + v^3 + y + 5z + 12v^3 + x"
- (4, 2) = "3x^3 + 2z + 12x^3 + 7z"

