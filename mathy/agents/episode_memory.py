from multiprocessing import Queue
from typing import List, Optional
import numpy as np
from ..state import (
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)


def rnn_weighted_history(observations: List[MathyObservation], rnn_size: int = 128):
    # Build a historical LSTM state: https://arxiv.org/pdf/1810.04437.pdf
    #
    # Take the mean of the previous LSTM states for this episode, which has
    # contains weighted information where stored states that persisted have
    # weight proportional to the number of timesteps the LSTM gating mechanism
    # kept the value in its memory.
    if len(observations) > 0:
        in_h = []
        in_c = []
        for obs in observations:
            in_h.append(obs.rnn_state[0])
            in_c.append(obs.rnn_state[1])
        # Take the mean of the historical states:
        memory_context_h = np.array(in_h).mean(axis=0)
        memory_context_c = np.array(in_c).mean(axis=0)
    else:
        memory_context_h = np.zeros([1, rnn_size])
        memory_context_c = np.zeros([1, rnn_size])
    return [memory_context_h, memory_context_c]


class EpisodeMemory(object):
    # Observation from the environment
    observations: List[MathyObservation]
    # Action taken in the environment
    actions: List[int]
    # Reward from the environmnet
    rewards: List[float]
    # Estimated value from the model
    values: List[float]
    # Grouping Control error from the environment
    grouping_changes: List[float]

    def __init__(self):
        self.clear()

    @property
    def ready(self) -> bool:
        return len(self.observations) > 3

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.grouping_changes = []

    def to_window_observation(
        self, observation: MathyObservation, window_size: int = 1
    ) -> MathyWindowObservation:
        previous = -(window_size - 1)
        window_observations = self.observations[previous:] + [observation]
        return observations_to_window(window_observations)

    def to_episode_window(self) -> MathyWindowObservation:
        return observations_to_window(self.observations)

    def store(
        self,
        *,
        observation: MathyObservation,
        action: int,
        reward: float,
        grouping_change: float,
        value: float,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.grouping_changes.append(grouping_change)

    def rnn_weighted_history(self, rnn_size):
        return rnn_weighted_history(self.observations, rnn_size)
