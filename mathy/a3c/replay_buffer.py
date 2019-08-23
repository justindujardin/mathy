import tensorflow as tf
from typing import List, Any, Tuple
from ..core.expressions import MathTypeKeys
from ..agent.features import (
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


class ReplayBuffer(object):
    states: List[Any] = []
    actions: List[int] = []
    rewards: List[float] = []

    def __init__(self):
        self.states = [True]
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def to_features(self):
        context_feature_keys: List[str] = [
            FEATURE_LAST_RULE,
            FEATURE_NODE_COUNT,
            FEATURE_MOVE_COUNTER,
            FEATURE_MOVES_REMAINING,
            FEATURE_PROBLEM_TYPE,
        ]
        sequence_feature_keys: List[Tuple[str, bool]] = [
            (FEATURE_FWD_VECTORS, False),
            (FEATURE_BWD_VECTORS, True),
            (FEATURE_LAST_FWD_VECTORS, False),
            (FEATURE_LAST_BWD_VECTORS, True),
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
        lengths = [len(s[FEATURE_BWD_VECTORS][0]) for s in self.states]
        max_sequence = max(lengths)
        for key in context_feature_keys:
            for state in self.states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(tf.convert_to_tensor(state[key]))
        for key, backward in sequence_feature_keys:
            pad_value = tuple([MathTypeKeys["empty"]] * 3)
            for state in self.states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(
                    pad_array(
                        state[key][0], max_sequence, pad_value, backwards=backward
                    )
                )

        max_sequence = max([len(s[FEATURE_MOVE_MASK][0]) for s in self.states])
        for state in self.states:
            # Max length of move masks in batch
            padded = pad_array(state[FEATURE_MOVE_MASK][0], max_sequence, 0.0)
            out[FEATURE_MOVE_MASK].append(
                tf.convert_to_tensor(padded, dtype=tf.float32)
            )

        # assert batch length in move masks
        _check_mask = -1
        for mask in out[FEATURE_MOVE_MASK]:
            if _check_mask == -1:
                _check_mask = len(mask)
            assert (
                len(mask) == _check_mask
            ), f"Expected same length, but for {len(mask)} and {_check_mask}"
        return out

