# mathy.agent.util

## EpisodeLosses <kbd>dataclass</kbd>
```python
EpisodeLosses(self, data: Dict[str, float] = <factory>) -> None
```
Store a set of losses keyed by a string that is used when printing them
## record <kbd>function</kbd>
```python
record(
    episode: int, 
    is_win: bool, 
    episode_reward: float, 
    worker_idx: int, 
    global_ep_reward: float, 
    losses: mathy.agent.util.EpisodeLosses, 
    num_steps: int, 
    env_name: str, 
    env: mathy.env.MathyEnv, 
    state: mathy.state.MathyEnvState, 
)
```
Helper function to store score and print statistics.

__Arguments__


- __episode__: Current episode

- __episode_reward__: Reward accumulated over the current episode

- __worker_idx__: Which thread (worker)

- __global_ep_reward__: The moving average of the global reward

- __total_loss__: The total loss accumualted over the current episode

- __num_steps__: The number of steps the episode took to complete

- __env_name__: The environment name for the episode

## truncate <kbd>function</kbd>
```python
truncate(value: Union[str, int, float])
```
Truncate a number to 3 decimal places
