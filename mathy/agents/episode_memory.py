from multiprocessing import Queue
from typing import List, Optional
import numpy as np
from ..state import (
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from .experience import Experience, ExperienceFrame


class EpisodeMemory(object):
    # Observation from the environment
    observations: List[MathyObservation]
    # Action taken in the environment
    actions: List[int]
    # Reward from the environmnet
    rewards: List[float]
    # Estimated value from the model
    values: List[float]
    # Experience frame for replay
    frames: List[ExperienceFrame]
    # Grouping Control error from the environment
    grouping_changes: List[float]

    def __init__(
        self, experience: Optional[Experience] = None, exp_out: Optional[Queue] = None
    ):
        self.experience = experience
        self.exp_out = exp_out
        self.clear()

    @property
    def ready(self) -> bool:
        return len(self.observations) > 3

    def clear(self):
        self.observations = []
        self.frames = []
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
        frame: ExperienceFrame,
        value: float,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.frames.append(frame)
        self.values.append(value)
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
        if self.experience is not None:
            for f in self.frames:
                self.experience.add_frame(f)
        self.frames = []

    def rnn_weighted_history(self, rnn_size):
        # Build a historical LSTM state: https://arxiv.org/pdf/1810.04437.pdf
        #
        # Take the mean of the previous LSTM states for this episode, which has
        # contains weighted information where stored states that persisted have
        # weight proportional to the number of timesteps the LSTM gating mechanism
        # kept the value in its memory.
        if len(self.observations) > 0:
            in_h = []
            in_c = []
            for obs in self.observations:
                in_h.append(obs.rnn_state[0])
                in_c.append(obs.rnn_state[1])
            # Take the mean of the historical states:
            memory_context_h = np.array(in_h).mean(axis=0)
            memory_context_c = np.array(in_c).mean(axis=0)
        else:
            memory_context_h = np.zeros([1, rnn_size])
            memory_context_c = np.zeros([1, rnn_size])
        return [memory_context_h, memory_context_c]
