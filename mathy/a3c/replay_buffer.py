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
        context_feature_keys: List[str] = [FEATURE_LAST_RULE, FEATURE_NODE_COUNT]
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
        }
        for key in context_feature_keys:
            for state in self.states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(tf.convert_to_tensor(state[key]))
        for key, backward in sequence_feature_keys:
            lengths = [len(s[key][0]) for s in self.states]
            max_sequence = max(lengths)
            pad_value = tuple([MathTypeKeys["empty"]] * 3)
            for state in self.states:
                if key not in state:
                    raise ValueError(f"key '{key}' not found in state: {state}'")
                out[key].append(
                    pad_array(
                        state[key][0], max_sequence, pad_value, backwards=backward
                    )
                )

        return out

