from multiprocessing import Queue
from typing import List, Optional

from ...state import (
    MathyBatchObservation,
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
    windows_to_batch,
)
from .experience import Experience, ExperienceFrame


class EpisodeMemory(object):
    observations: List[MathyObservation]
    actions: List[int]
    rewards: List[float]
    frames: List[ExperienceFrame]
    grouping_changes: List[float]

    def __init__(self, experience: Experience, exp_out: Optional[Queue] = None):
        self.experience = experience
        self.exp_out = exp_out
        self.observations = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.grouping_changes = []

    @property
    def ready(self) -> bool:
        return len(self.observations) > 3

    def get_current_batch(
        self, current_observation: MathyObservation, mathy_state: MathyEnvState
    ) -> MathyBatchObservation:
        observations = self.observations[-2:] + [current_observation]
        rnn_size = len(current_observation.rnn_state[0])
        while len(observations) < 3:
            observations.insert(0, mathy_state.to_empty_observation(rnn_size=rnn_size))
        return windows_to_batch([observations_to_window(observations)])

    def to_window_observation(
        self, observation: MathyObservation, window_size: int = 2
    ) -> MathyWindowObservation:
        previous = -(window_size - 1)
        window_observations = self.observations[previous:] + [observation]
        return observations_to_window(window_observations)

    def to_episode_window(self) -> MathyWindowObservation:
        return observations_to_window(self.observations)

    def store(
        self,
        state,
        action: int,
        reward: float,
        grouping_change: float,
        frame: ExperienceFrame,
    ):
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.frames.append(frame)
        self.grouping_changes.append(grouping_change)

    def commit_frames(
        self, worker_index: int, ground_truth_discounted_rewards: List[float]
    ):
        """Commit frames to the replay buffer when a terminal state is reached
        so that we have ground truth rewards rather than predictions for value
        replay """
        if self.exp_out is None:
            raise ValueError(
                "espisode memory was not given a queue to submit observations to"
            )
        # Ensure the data is the right shape
        assert len(ground_truth_discounted_rewards) == len(
            self.frames
        ), "discounted rewards must be for all frames"
        for frame, reward in zip(self.frames, ground_truth_discounted_rewards):
            frame.reward = reward
        # share experience with other workers
        self.exp_out.put((worker_index, self.frames))
        for f in self.frames:
            self.experience.add_frame(f)
        self.frames = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.frames = []
        self.grouping_changes = []
