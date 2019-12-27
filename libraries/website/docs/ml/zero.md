## Zero

The MCTS and Neural Network powered (Zero) agent is inspired by the work of Google's DeepMind and their AlphZero board-game playing AI. It uses a Monte Carlo Tree Search algorithm to produce quality actions that are unbiased by things like Actor/Critic errors.

## Examples

### Training

Use the self-play functionality to train a zero agent using the [Policy/Value](/ml/policy_value) model:

```python
{!./snippets/ml/zero_training.py!}
```
