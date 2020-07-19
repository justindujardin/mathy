# mathy.agent.util

## EpisodeLosses
```python
EpisodeLosses(self, data:Dict[str, float]=<factory>) -> None
```
Store a set of losses keyed by a string that is used when printing them
## record
```python
record(
    episode: int,
    is_win: bool,
    episode_reward: float,
    worker_idx: int,
    global_ep_reward: float,
    result_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x10ccc5860>>,
    losses: mathy.agent.util.EpisodeLosses,
    num_steps: int,
    env_name: str,
)
```
Helper function to store score and print statistics.
Arguments:
  episode: Current episode
  episode_reward: Reward accumulated over the current episode
  worker_idx: Which thread (worker)
  global_ep_reward: The moving average of the global reward
  result_queue: Queue storing the moving average of the scores
  total_loss: The total loss accumualted over the current episode
  num_steps: The number of steps the episode took to complete

