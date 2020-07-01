from typing import List, Tuple

import numpy as np
import tensorflow as tf

from ..state import MathyEnvState, MathyInputsType, MathyWindowObservation
from .model import AgentModel


def predict_next(
    model: AgentModel, inputs: MathyInputsType
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Predict one probability distribution and value for the
    given sequence of inputs """
    logits, values, masked, rewards = model.call(inputs)
    # take the last timestep
    masked = masked[-1][:]
    flat_logits = tf.reshape(tf.squeeze(masked), [-1])
    probs = tf.nn.softmax(flat_logits)
    return probs, tf.squeeze(values[-1])


class ActionSelector:
    """An episode-specific selector of actions"""

    model: AgentModel
    worker_id: int
    episode: int

    def __init__(self, *, model: AgentModel, episode: int, worker_id: int):
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
        probs, value = predict_next(self.model, last_window.to_inputs())
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

        probs, value = predict_next(self.model, last_window.to_inputs())
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
