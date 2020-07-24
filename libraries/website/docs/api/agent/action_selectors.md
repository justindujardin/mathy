# mathy.agent.action_selectors

## ActionSelector
```python
ActionSelector(
    self,
    model: tensorflow.python.keras.engine.training.Model,
    episode: int,
    worker_id: int,
)
```
An episode-specific selector of actions
## apply_pi_mask
```python
apply_pi_mask(
    logits: tensorflow.python.framework.ops.Tensor,
    mask: tensorflow.python.framework.ops.Tensor,
    predictions: int,
) -> tensorflow.python.framework.ops.Tensor
```
Take the policy_mask from a batch of features and multiply
the policy logits by it to remove any invalid moves
## predict_next
```python
predict_next(
    model: tensorflow.python.keras.engine.training.Model,
    inputs: Dict[str, Any],
) -> Tuple[tensorflow.python.framework.ops.Tensor, tensorflow.python.framework.ops.Tensor]
```
Predict one probability distribution and value for the
given sequence of inputs
