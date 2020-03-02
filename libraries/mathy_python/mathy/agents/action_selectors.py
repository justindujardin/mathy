from typing import List, Tuple

import numpy as np

from ..state import MathyEnvState, MathyWindowObservation
from .policy_value_model import PolicyValueModel
from .mcts import MCTS


class ActionSelector:
    """An episode-specific selector of actions"""

    model: PolicyValueModel
    worker_id: int
    episode: int

    def __init__(self, *, model: PolicyValueModel, episode: int, worker_id: int):
        self.model = model
        self.worker_id = worker_id
        self.episode = episode

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
    ) -> Tuple[int, float]:
        raise NotImplementedError(self.select)


class GreedyActionSelector(ActionSelector):
    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
    ) -> Tuple[int, float]:
        probs, value = self.model.predict_next(last_window.to_inputs())
        action = np.argmax(probs)
        return action, float(value)


class A3CEpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, *, epsilon: float, **kwargs):
        super(A3CEpsilonGreedyActionSelector, self).__init__(**kwargs)
        self.epsilon = epsilon

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
    ) -> Tuple[int, float]:

        probs, value = self.model.predict_next(last_window.to_inputs())
        last_move_mask = last_window.mask[-1]
        no_random = bool(self.worker_id == 0)
        if not no_random and np.random.random() < self.epsilon:
            # Select a random action
            action_mask = last_move_mask[:]
            # normalize all valid actions to equal probability
            actions = action_mask / np.sum(action_mask)
            action = np.random.choice(len(actions), p=actions)
        else:
            action = np.argmax(probs)
        return action, float(value)


class MCTSActionSelector(ActionSelector):
    def __init__(self, mcts: MCTS, **kwargs):
        super(MCTSActionSelector, self).__init__(**kwargs)
        self.mcts = mcts

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
    ) -> Tuple[int, float]:
        probs, value = self.mcts.estimate_policy(last_state)
        return np.argmax(probs), value
