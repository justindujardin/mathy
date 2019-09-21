import math
from multiprocessing import Queue
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..core.expressions import MathTypeKeys
from ..features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVE_MASK,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    pad_array,
)
from ..state import (
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    MathyBatchObservation,
    observations_to_window,
    windows_to_batch,
)
from ..util import GameRewards
from .experience import Experience, ExperienceFrame


class EpisodeMemory(object):
    observations: List[MathyObservation]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    frames: List[ExperienceFrame]
    grouping_changes: List[float]

    def __init__(self, experience_queue: Optional[Queue] = None):
        self.experience_queue = experience_queue
        self.observations = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.grouping_changes = []

    @property
    def ready(self) -> bool:
        return len(self.observations) > 3

    def get_current_window(
        self, current_observation: MathyObservation, mathy_state: MathyEnvState
    ):
        observations = self.observations[-2:] + [current_observation]
        while len(observations) < 3:
            observations.insert(0, mathy_state.to_empty_observation())
        return windows_to_batch([observations_to_window(observations)])

    def store(
        self,
        state,
        action: int,
        reward: float,
        value: float,
        grouping_change: float,
        frame: ExperienceFrame,
    ):
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.frames.append(frame)
        self.grouping_changes.append(grouping_change)

    def commit_frames(self, ground_truth_discounted_rewards: List[float]):
        """Commit frames to the replay buffer when a terminal state is reached
        so that we have ground truth rewards rather than predictions for value
        replay """
        if self.experience_queue is None:
            raise ValueError(
                "espisode memory was not given a queue to submit observations to"
            )
        # Ensure the data is the right shape
        assert len(ground_truth_discounted_rewards) == len(
            self.frames
        ), "discounted rewards must be for all frames"
        for frame, reward in zip(self.frames, ground_truth_discounted_rewards):
            frame.reward = reward
            self.experience_queue.put(frame)
        self.frames = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.frames = []
        self.grouping_changes = []
