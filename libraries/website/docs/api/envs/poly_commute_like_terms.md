# mathy.envs.poly_commute_like_terms

## PolyCommuteLikeTerms
```python
PolyCommuteLikeTerms(self, **kwargs)
```
A Mathy environment for moving like terms near each other to enable
further simplification.

This task is intended to test the model's ability to identify like terms
in a large string of unlike terms and its ability to use the commutative
swap rule to reorder the expression bringing the like terms close together.

### transition_fn
```python
PolyCommuteLikeTerms.transition_fn(self, env_state:mathy.state.MathyEnvState, expression:mathy.core.expressions.MathExpression, features:mathy.state.MathyObservation) -> Union[tf_agents.trajectories.time_step.TimeStep, NoneType]
```
If the expression has any nodes that the DistributiveFactorOut rule
can be applied to, the problem is solved.
### max_moves_fn
```python
PolyCommuteLikeTerms.max_moves_fn(self, problem:mathy.types.MathyEnvProblem, config:mathy.types.MathyEnvProblemArgs) -> int
```
This task is to move two terms near each other, which requires
as many actions as there are blocker nodes. The problem complexity
is a direct measure of this value.
