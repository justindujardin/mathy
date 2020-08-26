from typing import List, Tuple

import numpy as np
import tensorflow as tf

from ..state import MathyEnvState, MathyInputsType, MathyWindowObservation
from .model import AgentModel, call_model


# def apply_pi_mask(
#     action_logits: tf.Tensor, args_logits: tf.Tensor, mask: tf.Tensor
# ) -> Tuple[tf.Tensor, tf.Tensor]:
#     return fn_masked, args_masked


def predict_next(
    model: AgentModel, inputs: MathyInputsType
) -> Tuple[Tuple[int, int], float]:
    """Predict the fn/args policies and value/reward estimates for current timestep."""
    mask = inputs.pop("mask_in")[-1]
    action_logits, args_logits, values = call_model(model, inputs)
    action_logits = action_logits[-1:]
    args_logits = args_logits[-1:]
    values = values[-1:]
    fn_mask = mask.numpy().sum(axis=1).astype("bool").astype("int")
    fn_mask_logits = tf.multiply(action_logits, fn_mask, name="mask_logits")
    # Mask the selected rule functions to remove invalid selections
    fn_masked = tf.where(
        tf.equal(fn_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(action_logits), -1000000.0),
        fn_mask_logits,
        name="fn_softmax_negative_logits",
    )
    fn_probs = tf.nn.softmax(fn_masked).numpy()
    fn_action = int(fn_probs.argmax())
    # Mask the node selection policy for each rule function
    args_mask = tf.cast(mask[fn_action], dtype="float32")
    args_mask_logits = tf.multiply(args_logits, args_mask, name="mask_logits")
    args_masked = tf.where(
        tf.equal(args_mask_logits, tf.constant(0.0)),
        tf.fill(tf.shape(args_logits), -1000000.0),
        args_mask_logits,
        name="args_negative_softmax_logts",
    )
    args_probs = tf.nn.softmax(args_masked).numpy()
    arg_action = int(args_probs.argmax())
    return (fn_action, arg_action), float(tf.squeeze(values[-1]).numpy())


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
        return predict_next(self.model, last_window.to_inputs())


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

        return predict_next(self.model, last_window.to_inputs())
