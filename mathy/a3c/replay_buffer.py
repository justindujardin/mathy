import tensorflow as tf
import math
from ..util import GameRewards
from typing import List, Any, Tuple, Optional
from ..core.expressions import MathTypeKeys
from ..features import (
    pad_array,
    FEATURE_LAST_RULE,
    FEATURE_NODE_COUNT,
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_MOVE_MASK,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_PROBLEM_TYPE,
)
from .experience import ExperienceFrame, Experience
import numpy as np


class ReplayBuffer(object):
    states: List[Any]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    frames: List[ExperienceFrame]
    grouping_changes: List[float]

    def __init__(self, experience: Experience):
        self.experience = experience
        self.states = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.grouping_changes = []

    @property
    def ready(self) -> bool:
        return len(self.states) > 3

    def get_current_window(self, current_state):
        window_states = self.states[-2:] + [current_state]
        return self.to_features(window_states)

    def store(
        self,
        state,
        action: int,
        reward: float,
        value: float,
        grouping_change: float,
        frame: ExperienceFrame,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.frames.append(frame)
        self.grouping_changes.append(grouping_change)

    def commit_frames(self, ground_truth_discounted_rewards: List[float]):
        """Commit frames to the replay buffer when a terminal state is reached
        so that we have ground truth rewards rather than predictions for value
        replay"""
        # Ensure the data is the right shape
        assert len(ground_truth_discounted_rewards) == len(
            self.frames
        ), "discounted rewards must be for all frames"
        for frame, reward in zip(self.frames, ground_truth_discounted_rewards):
            frame.reward = reward
            self.experience.add_frame(frame)
        self.frames = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.frames = []
        self.grouping_changes = []

    def rp_samples(self, max_samples=2) -> Tuple[List[Any], List[float]]:
        outputs: List[List[ExperienceFrame]] = []
        rewards: List[float] = []
        if self.experience.is_full() is False:
            return outputs, rewards
        for i in range(max_samples):
            frames = self.experience.sample_rp_sequence()
            # 4 frames
            states = [frame.state for frame in frames[:-1]]
            sample_features = self.to_features(states)
            target_reward = frames[-1].reward
            if math.isclose(target_reward, GameRewards.TIMESTEP):
                sample_label = 0  # zero
            elif target_reward > 0:
                sample_label = 1  # positive
            else:
                sample_label = 2  # negative
            outputs.append(sample_features)
            rewards.append(sample_label)
        return outputs, rewards

    def to_features(self, feature_states: Optional[List[Any]] = None) -> List[Any]:
        if feature_states is None:
            feature_states = self.states
        context_feature_keys: List[str] = [
            FEATURE_LAST_RULE,
            FEATURE_NODE_COUNT,
            FEATURE_MOVE_COUNTER,
            FEATURE_MOVES_REMAINING,
        ]
        text_feature_keys: List[str] = [FEATURE_PROBLEM_TYPE]
        sequence_feature_keys: List[Tuple[str, bool]] = [
            (FEATURE_FWD_VECTORS, False),
            (FEATURE_BWD_VECTORS, True),
        ]
        out = {
            FEATURE_LAST_RULE: [],
            FEATURE_NODE_COUNT: [],
            FEATURE_FWD_VECTORS: [],
            FEATURE_BWD_VECTORS: [],
            FEATURE_LAST_FWD_VECTORS: [],
            FEATURE_LAST_BWD_VECTORS: [],
            FEATURE_MOVE_MASK: [],
            FEATURE_MOVE_COUNTER: [],
            FEATURE_MOVES_REMAINING: [],
            FEATURE_PROBLEM_TYPE: [],
        }
        lengths = [len(s[FEATURE_BWD_VECTORS][0]) for s in feature_states]
        max_sequence = max(lengths)
        for key in context_feature_keys:
            for state in feature_states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(tf.convert_to_tensor(state[key]))

        for key in text_feature_keys:
            for state in feature_states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(state[key][0])

        # last_size = -1
        for key, backward in sequence_feature_keys:
            pad_value = [MathTypeKeys["empty"]]
            for state in feature_states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                src_seq = state[key][0]
                new_seq = pad_array(
                    src_seq, max_sequence, pad_value, backwards=backward, cleanup=True
                )
                # new_size = np.array(new_seq).size
                # if last_size != -1 and new_size != last_size:
                #     # Print out the failing example states
                #     for s in feature_states:
                #         print(s)
                #     raise ValueError("Padded arrays have different sizes!")
                out[key].append(new_seq)

        max_sequence = max([len(s[FEATURE_MOVE_MASK][0]) for s in feature_states])
        for state in feature_states:
            # Max length of move masks in batch
            padded = pad_array(
                state[FEATURE_MOVE_MASK][0], max_sequence, 0.0, cleanup=True
            )
            out[FEATURE_MOVE_MASK].append(
                tf.convert_to_tensor(padded, dtype=tf.float32)
            )

        # # assert batch length in move masks
        # _check_mask = -1
        # for mask in out[FEATURE_MOVE_MASK]:
        #     if _check_mask == -1:
        #         _check_mask = len(mask)
        #     assert (
        #         len(mask) == _check_mask
        #     ), f"Expected same length, but for {len(mask)} and {_check_mask}"
        return out

