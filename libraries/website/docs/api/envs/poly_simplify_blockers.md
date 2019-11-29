# mathy.envs.poly_simplify_blockers

## PolySimplifyBlockers
```python
PolySimplifyBlockers(self, rules=None, rewarding_actions=None, max_moves=20, verbose=False, reward_discount=0.99)
```
A Mathy environment for polynomial problems that have a variable
string of mismatched terms separating two like terms.

The goal is to:
  1. Commute the like terms so they become siblings
  2. Combine the sibling like terms

