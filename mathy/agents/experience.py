"""Implementation adapted from miyosuda/unreal:

https://github.com/miyosuda/unreal/tree/master/train"""
from collections import deque
from typing import Any, Optional

import numpy as np

from ..gym import MathyGymEnv
from ..util import GameRewards


class ExperienceFrame(object):

    # The state of the frame
    state: Any
    # The action taken at "state"
    action: int
    # The reward given for the action taken at "state"
    reward: float

    # The action taken that resulted in a transition to "state"
    last_action: int
    # The reward received for taking the action that transitioned to "state"
    last_reward: float

    # Whether or not the action resulted in a terminal state
    terminal: bool
    # The final discounted reward value assigned after the episode is complete
    discounted: float

    # The expression grouping change that occurred transitioning to "state"
    grouping_change: float

    # RNN state before transition to "state"
    rnn_state: np.ndarray

    def __init__(
        self,
        *,
        state: Any,
        reward: float,
        action: int,
        terminal: bool,
        discounted: float = -1.0,
        grouping_change: float,
        last_action: int,
        last_reward: float,
        rnn_state: np.ndarray
    ):
        self.state = state
        self.action = action
        self.reward = np.clip(reward, -1, 1)
        self.terminal = terminal
        self.grouping_change = grouping_change
        self.last_action = last_action
        self.last_reward = np.clip(last_reward, -1, 1)
        self.discounted = discounted
        self.rnn_state = rnn_state

    def get_last_action_reward(self, action_size):
        """Return one hot vectored last action + last reward."""
        return ExperienceFrame.concat_action_and_reward(
            self.last_action, action_size, self.last_reward
        )

    def get_action_reward(self, action_size):
        """Return one hot vectored action + reward."""
        return ExperienceFrame.concat_action_and_reward(
            self.action, action_size, self.reward
        )

    @staticmethod
    def concat_action_and_reward(action, action_size, reward):
        """Return one hot vectored action and reward."""
        action_reward = np.zeros([action_size + 1])
        action_reward[action] = 1.0
        action_reward[-1] = float(reward)
        return action_reward


class Experience(object):
    def __init__(self, history_size: int, ready_at: int = None):
        self._ready_at = history_size if ready_at is None else ready_at
        self._history_size = history_size
        self._frames: deque = deque(maxlen=history_size)
        # frame indices for zero rewards
        self._zero_reward_indices: deque = deque()
        # frame indices for non zero rewards
        self._non_zero_reward_indices: deque = deque()
        self._top_frame_index = 0

    def add_frame(self, frame: ExperienceFrame):
        if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
            # Discard if terminal frame continues
            return

        frame_index = self._top_frame_index + len(self._frames)
        was_full = self.is_full()

        # append frame
        self._frames.append(frame)

        # append index if there are enough (because we replay 4 at a time)
        if frame_index >= 3:
            # UNREAL uses 0 or non-zero, but we have a penatly timestep, so
            # consider anything less than that to be zero.
            if frame.reward <= GameRewards.TIMESTEP:
                self._zero_reward_indices.append(frame_index)
            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + 3
            # Cut frame if its index is lower than cut_frame_index.
            if (
                len(self._zero_reward_indices) > 0
                and self._zero_reward_indices[0] < cut_frame_index
            ):
                self._zero_reward_indices.popleft()

            if (
                len(self._non_zero_reward_indices) > 0
                and self._non_zero_reward_indices[0] < cut_frame_index
            ):
                self._non_zero_reward_indices.popleft()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def is_full(self):
        return len(self._frames) >= self._ready_at

    def sample_sequence(
        self, sequence_size: int, fill_from_env: Optional[MathyGymEnv] = None
    ):
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        curr_size = len(self._frames)
        start_pos = np.random.randint(0, curr_size - sequence_size - 1)

        if self._frames[start_pos].terminal:
            start_pos += 1
            # Assuming that there are no successive terminal frames.

        sampled_frames = []

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_frames.append(frame)
            if frame.terminal:
                break

        return sampled_frames

    def sample_rp_sequence(self):
        """Sample 4 successive frames for reward prediction."""
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self._zero_reward_indices) == 0:
            # zero rewards container was empty
            from_zero = False
        elif len(self._non_zero_reward_indices) == 0:
            # non zero rewards container was empty
            from_zero = True

        if from_zero:
            index = np.random.randint(len(self._zero_reward_indices))
            end_frame_index = self._zero_reward_indices[index]
        else:
            index = np.random.randint(len(self._non_zero_reward_indices))
            end_frame_index = self._non_zero_reward_indices[index]

        start_frame_index = end_frame_index - 3
        raw_start_frame_index = start_frame_index - self._top_frame_index

        sampled_frames = []

        for i in range(4):
            frame = self._frames[raw_start_frame_index + i]
            sampled_frames.append(frame)

        return sampled_frames
