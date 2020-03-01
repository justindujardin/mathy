import datetime
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict

from colr import color


@dataclass
class EpisodeLosses:
    """Store a set of losses keyed by a string that is used when printing them"""

    data: Dict[str, float] = field(default_factory=dict)

    def reset(self):
        self.data = dict()

    def increment(self, key: str, value: float) -> None:
        if not key in self.data:
            self.data[key] = 0.0
        self.data[key] += value

    def format_loss(self, key: str, value: float):
        return "{}: {:<8}".format(key, truncate(value))

    def __str__(self):
        out = ""
        for key, value in self.data.items():
            out += f"{self.format_loss(key, value)} "
        return out


def truncate(value):
    return float("%.3f" % (float(value)))


def record(
    episode: int,
    is_win: bool,
    episode_reward: float,
    worker_idx: int,
    global_ep_reward: float,
    result_queue: multiprocessing.Queue,
    losses: EpisodeLosses,
    num_steps: int,
    env_name: str,
):
    """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """

    now = datetime.datetime.now().strftime("%H:%M:%S")
    # Clamp to range -2, 2
    episode_reward = min(2.0, max(-2.0, episode_reward))

    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01

    fore = "green"
    if not is_win:
        fore = "red"
    elif episode_reward < 1.0:
        fore = "yellow"
    heading = "{:<8} {:<3} {:<8} {:<10}".format(
        now, f"w{worker_idx}", f"ep: {episode}", f"steps: {num_steps}"
    )
    rewards = "r_avg: {:<6} r_ep: {:<6}".format(
        truncate(global_ep_reward), truncate(episode_reward)
    )
    print(
        color(f"{heading} {rewards} {losses} [{env_name}]", fore=fore, style="bright")
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
