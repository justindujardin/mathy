from typing import List, Tuple

import numpy as np
import tensorflow as tf

from ..state import MathyEnvState, MathyInputsType, MathyWindowObservation
from .model import AgentModel, call_model


def apply_pi_mask(
    action_logits: tf.Tensor, args_logits: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    fn_mask = mask.numpy().sum(axis=2).astype("bool").astype("int")
    fn_mask_logits = tf.multiply(action_logits, fn_mask, name="mask_logits")
    # Mask the selected rule functions to remove invalid selections
    fn_masked = tf.where(
        tf.equal(fn_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(action_logits), -1000000.0),
        fn_mask_logits,
        name="fn_softmax_negative_logits",
    )
    # Mask the node selection policy for each rule function
    args_mask = tf.cast(mask, dtype="float32")

    args_logits = args_logits[:, :, : tf.shape(mask)[-1]]
    args_mask_logits = tf.multiply(args_logits, args_mask, name="mask_logits")
    args_masked = tf.where(
        tf.equal(args_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(args_logits), -1000000.0),
        args_mask_logits,
        name="args_negative_softmax_logts",
    )
    return fn_masked, args_masked


def predict_next(
    model: AgentModel, inputs: MathyInputsType
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Predict the fn/args policies and value/reward estimates for current timestep."""
    mask = inputs.pop("mask_in")
    logits, params, values, rewards = call_model(model, inputs)
    fn_masked, args_masked = apply_pi_mask(logits, params, mask)
    fn_masked = fn_masked[-1][:]
    fn_probs = tf.nn.softmax(fn_masked).numpy().tolist()
    args_masked = args_masked[-1][:]
    args_probs = tf.nn.softmax(args_masked).numpy().tolist()
    return fn_probs, args_probs, tf.squeeze(values[-1]), tf.squeeze(rewards[-1])


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
    ) -> Tuple[Tuple[int, int], float]:
        raise NotImplementedError(self.select)


class GreedyActionSelector(ActionSelector):
    def select(
        self,
        *,
        last_state: MathyEnvState,
        last_window: MathyWindowObservation,
        last_action: int,
        last_reward: float,
    ) -> Tuple[Tuple[int, int], float]:
        fn_probs, args_probs, value, reward = predict_next(
            self.model, last_window.to_inputs()
        )
        fn_action = int(np.argmax(fn_probs))
        args_action = int(np.argmax(args_probs[fn_action]))
        return (fn_action, args_action), float(value)


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
    ) -> Tuple[Tuple[int, int], float]:

        fn_probs, args_probs, value, reward = predict_next(
            self.model, last_window.to_inputs()
        )
        no_random = self.worker_id == 0
        if not no_random and np.random.random() < self.epsilon:
            last_move_mask = last_window.mask[-1]
            # Select a random action
            action_mask = last_move_mask[:]
            valid_rules = [
                i for i, a in enumerate(action_mask) if (np.array(a) > 0).any()
            ]
            fn_action = np.random.choice(valid_rules)
            args_action = np.random.choice(np.nonzero(action_mask[fn_action])[0])
        else:
            fn_action = int(np.argmax(fn_probs))
            args_action = int(np.argmax(args_probs[fn_action]))
        return (fn_action, args_action), float(value)
