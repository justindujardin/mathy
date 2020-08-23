# mathy.agent.action_selectors

## ActionSelector <kbd>class</kbd>
```python
ActionSelector(
    self, 
    model: tensorflow.python.keras.engine.training.Model, 
    episode: int, 
    worker_id: int, 
)
```
An episode-specific selector of actions
## predict_next <kbd>function</kbd>
```python
predict_next(
    model: tensorflow.python.keras.engine.training.Model, 
    inputs: Dict[str, Any], 
) -> Tuple[tensorflow.python.framework.ops.Tensor, tensorflow.python.framework.ops.Tensor, tensorflow.python.framework.ops.Tensor, tensorflow.python.framework.ops.Tensor]
```
Predict the fn/args policies and value/reward estimates for current timestep.
