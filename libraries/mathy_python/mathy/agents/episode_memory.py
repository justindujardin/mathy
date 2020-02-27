from multiprocessing import Queue
from typing import Dict, List, Optional

import numpy as np

from ..state import (
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)


class EpisodeMemory(object):
    # Observation from the environment
    observations: List[MathyObservation]
    # Action taken in the environment
    actions: List[int]
    # Reward from the environmnet
    rewards: List[float]
    # Estimated value from the model
    values: List[float]

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []

    def clear_except_window(self, window_size: int):
        """Clear all data except for a window's worth of elements"""
        self.observations = self.observations[-window_size:]
        self.actions = self.actions[-window_size:]
        self.rewards = self.rewards[-window_size:]
        self.values = self.values[-window_size:]

    def to_window_observation(
        self, observation: MathyObservation, window_size: int = 3
    ) -> MathyWindowObservation:
        previous = -(max(window_size - 1, 1))
        window_observations = self.observations[previous:] + [observation]
        assert len(window_observations) <= window_size
        return observations_to_window(window_observations)

    def to_episode_window(self) -> MathyWindowObservation:
        return observations_to_window(self.observations)

    def store(
        self,
        *,
        observation: MathyObservation,
        action: int,
        reward: float,
        value: float,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
