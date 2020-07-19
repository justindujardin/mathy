from multiprocessing import Queue
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator

import numpy as np

from ..types import Literal

from ..state import (
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)


EpisodeMemoryKeys = Union[
    Literal["actions"], Literal["rewards"], Literal["values"],
]


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

    def to_window_observations(
        self,
        window: int = 3,
        stride: int = 1,
        other_keys: List[EpisodeMemoryKeys] = [],
        only_full_windows: bool = True,
        zip_with: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, ...]]:
        if zip_with is not None:
            o_len = len(self.observations)
            z_len = len(zip_with)
            assert (
                o_len == z_len
            ), f"zip_with len({z_len}) must match observations({o_len})"
        results = []
        for i in range(len(self.observations)):
            if i % stride != 0:
                continue
            start = i
            end = i + window
            i_window = observations_to_window(self.observations[start:end])
            # Maybe exclude partial windows
            if only_full_windows and len(i_window.nodes) != window:
                continue

            item: List[Any] = [i_window]
            for key in other_keys:
                item += [getattr(self, key)[start:end]]
            if zip_with is not None:
                item += [zip_with[start:end]]
            results.append(tuple(item))
        return results

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
