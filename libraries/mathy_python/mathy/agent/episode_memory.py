from multiprocessing import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from mathy_envs.state import MathyObservation, MathyWindowObservation, observations_to_window
from mathy_envs.types import ActionList, ActionType, RewardList
from ..types import Literal

EpisodeMemoryKeys = Union[
    Literal["actions"], Literal["rewards"], Literal["values"],
]


class EpisodeMemory(object):
    # The max length of observations
    max_len: int
    # Observation from the environment
    observations: List[MathyObservation]
    # Action taken in the environment
    actions: List[ActionType]
    # Reward from the environmnet
    rewards: List[float]
    # Estimated value from the model
    values: List[float]

    def __init__(self, max_len: int, stack: int = 3):
        self.max_len = max_len
        self.stack = stack
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
        # Pad with the current observation so the model graph doesn't
        # need to be rebuilt for multiple batch sizes
        while len(window_observations) < window_size:
            window_observations.insert(0, MathyObservation.empty(observation))
        return observations_to_window(window_observations, self.max_len)

    def to_non_terminal_training_window(
        self, window_size: int = 3
    ) -> Tuple[MathyWindowObservation, ActionList, RewardList]:
        previous = -(max(window_size, 1))
        window_observations = self.observations[previous:]
        window_actions = self.actions[previous:]
        window_rewards = self.rewards[previous:]
        # Pad with the current observation so the model graph doesn't
        # need to be rebuilt for multiple batch sizes
        while len(window_observations) < window_size:
            window_observations.insert(0, MathyObservation.empty(self.observations[-1]))
            window_actions.insert(0, (0, 0))
            window_rewards.insert(0, 0.0)
        window = observations_to_window(window_observations, self.max_len)
        return window, window_actions, window_rewards

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
            i_window = observations_to_window(
                self.observations[start:end], self.max_len
            )
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
        window_observations = self.observations[:]
        # Pad with the current observation so the model graph doesn't
        # need to be rebuilt for multiple batch sizes
        while len(window_observations) < self.stack:
            window_observations.insert(
                0, MathyObservation.empty(window_observations[-1])
            )
        return observations_to_window(window_observations, self.max_len)

    def store(
        self,
        *,
        observation: MathyObservation,
        action: ActionType,
        reward: float,
        value: float,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
