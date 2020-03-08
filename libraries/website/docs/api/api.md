# mathy.api

## Mathy
```python
Mathy(
    self,
    model_path: str = None,
    model: mathy.agents.policy_value_model.PolicyValueModel = None,
    config: mathy.agents.base_config.BaseConfig = None,
    silent: bool = False,
)
```
The standard interface for working with Mathy models and agents.
### simplify
```python
Mathy.simplify(
    self,
    model: str = 'mathy_alpha_sm',
    problem: str,
    max_steps: int = 128,
) -> mathy.agents.episode_memory.EpisodeMemory
```
Simplify an input problem using the PolySimplify environment.

__Arguments__

- __model (str)__: The input model to use for picking simplifying actions
- __problem (str)__: The ascii math problem text, e.g. `-(4 + 2x) * 8 / 7y^(3 - 2)`
- __max_steps (int)__: The maximum number of episode steps to allow the agent to take
    while solving the problem. Taking more than this is considered a failure.

__Returns__

`(EpisodeMemory)`: The stored episode memory containing the intermediate steps to get
    to the solution for the input problem.


