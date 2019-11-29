# mathy.envs.binomial_distribute

## BinomialDistribute
```python
BinomialDistribute(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for distributing pairs of binomials.

The FOIL method is sometimes used to solve these types of problems, where
FOIL is just the distributive property applied to two binomials connected
with a multiplication.
### transition_fn
```python
BinomialDistribute.transition_fn(self, env_state:mathy.state.MathyEnvState, expression:mathy.core.expressions.MathExpression, features:mathy.state.MathyObservation) -> Union[tf_agents.trajectories.time_step.TimeStep, NoneType]
```
If there are no like terms.
### problem_fn
```python
BinomialDistribute.problem_fn(self, params:mathy.types.MathyEnvProblemArgs) -> mathy.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
2 binomials expressions connected by a multiplication.
