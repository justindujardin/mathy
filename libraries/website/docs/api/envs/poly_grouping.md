# mathy.envs.poly_grouping

## PolyGroupLikeTerms
```python
PolyGroupLikeTerms(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for grouping polynomial terms that are like.

The goal is to commute all the like terms so they become siblings as quickly as
possible.

### transition_fn
```python
PolyGroupLikeTerms.transition_fn(self, env_state:mathy.state.MathyEnvState, expression:mathy.core.expressions.MathExpression, features:mathy.state.MathyObservation) -> Union[tf_agents.trajectories.time_step.TimeStep, NoneType]
```
If all like terms are siblings.
