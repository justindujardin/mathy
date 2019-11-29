# mathy.envs.poly_combine_in_place

## PolyCombineInPlace
```python
PolyCombineInPlace(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for combining like terms in-place without
any commuting. This task is intended to test the model's ability
to identify like-terms among a bunch of unlike terms and combine
them with a sequence of two moves.

### max_moves_fn
```python
PolyCombineInPlace.max_moves_fn(self, problem:mathy.types.MathyEnvProblem, config:mathy.types.MathyEnvProblemArgs) -> int
```
When combining terms that are already siblings, we only need
to take two actions:

    1. distributive factor out the common element
    2. simplify the remaining constants


