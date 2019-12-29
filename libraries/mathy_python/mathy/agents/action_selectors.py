from typing import List, Tuple

import numpy as np

from ..state import MathyEnvState, MathyWindowObservation
from .policy_value_model import PolicyValueModel
from .mcts import MCTS


class ActionSelector:
    """An episode-specific selector of actions"""

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
        last_rnn_state: List[float],
    ) -> Tuple[int, float]:
        raise NotImplementedError(self.select)


class A3CEpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, *, epsilon: float, **kwargs):
        super(A3CEpsilonGreedyActionSelector, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.noise_alpha = 0.3
        self.use_noise = True
        self.first_step = True

    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
        last_rnn_state: List[float],
    ) -> Tuple[int, float]:

        probs, value = self.model.predict_next(last_window.to_inputs())
        last_move_mask = last_window.mask[-1]
        # Apply noise to the root node (like AlphaGoZero MCTS)
        if self.use_noise is True and self.first_step is True:
            noise = np.random.dirichlet([self.noise_alpha] * len(probs))
            noise = noise * np.array(last_move_mask)
            probs += noise
            pi_sum = np.sum(probs)
            if pi_sum > 0:
                probs /= pi_sum
            self.first_step = False

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
        last_rnn_state: List[float],
    ) -> Tuple[int, float]:
        probs, value = self.mcts.estimate_policy(last_state, last_rnn_state)
        return np.argmax(probs), value
