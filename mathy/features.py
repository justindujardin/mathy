import re
import numpy
import numpy as np
from typing import Tuple, Any, List, Dict
from .core import MathTypeKeys
from .state import MathyObservation
from enum import Enum
import math
from pydantic import BaseModel


def build_cnn_image_input(observation):
    """Build an image representation of the observation.

    Use the tree layout algorithm to draw a tree visually
    """
    pass


def calculate_chaos_node_control_signal(observation: MathyObservation):
    """node_ctrl signal is either 0 or 1 depending on if the input
    matches the output.

    So if a problem gets more of less complex (by adding or removing
    characters) then the resulting signal will be 0, but if nothing
    changes the signal is 1.

    This is intended not to prefer shortening or lengthening expressions
    but just change in the number of characters. The hope is that this
    will lead to a better representation that can deal with both the tasks
    of simplification and re-stating in more complex ways.

    Examples:
        "2x + 4x"      "4x + 2x"     = 7 == 7  = 1
        "2x + 4x"      "x * (2 + 4)" = 7 != 11 = 0
        "x * (2 + 4)"  "6x"          = 2 != 11 = 0
        "6x"           "x * (2 + 4)" = 2 != 11 = 0

    TODO: This might need to be a separately predicted policy, e.g.
            `pi2 = TimeDistributed(MathPolicyDropout)`
          Need to better understand comments on Q-learning being necessary

          https://arxiv.org/pdf/1611.05397.pdf

          "In principle, any reinforcement learning method could be applied to
           maximise these objectives. However, to efficiently learn to maximise
           many different pseudo-rewards simultaneously in parallel from a single
           stream of experience, it is necessary to use off-policy reinforcement
           learning. We focus on value-based RL methods that approximate the optimal
           action-values by Qlearning"
    """
    input = len(observation.input)
    output = len(observation.output)
    # lesser = min(input, output)
    # greater = max(input, output)
    # max protects against div by zero
    # signal = lesser / max(greater, 1)
    return 0 if input != output else 1


def calculate_brevity_node_control_signal(observation: MathyObservation):
    """node_ctrl signal is either 0 or 1 depending on if the output state
    has fewer nodes than the input. This doesn't always make sense, but for
    most problems in math I think it does. If it doesn't then expansion tends
    to make sense. I think perhaps these could switch based on the problem type.

    Examples:
        "2x + 4x"      "4x + 2x"     = 7 >= 7  = 0
        "2x + 4x"      "x * (2 + 4)" = 7 >= 11 = 1
        "x * (2 + 4)"  "6x"          = 11 >= 2 = 0
        "6x"           "x * (2 + 4)" = 2 >= 11 = 1

    TODO: This might need to be a separately predicted policy, e.g.
            `pi2 = TimeDistributed(MathPolicyDropout)`
          Need to better understand comments on Q-learning being necessary

          https://arxiv.org/pdf/1611.05397.pdf

          "In principle, any reinforcement learning method could be applied to
           maximise these objectives. However, to efficiently learn to maximise
           many different pseudo-rewards simultaneously in parallel from a single
           stream of experience, it is necessary to use off-policy reinforcement
           learning. We focus on value-based RL methods that approximate the optimal
           action-values by Qlearning"
    """
    input = len(observation.input)
    output = len(observation.output)
    return 0 if input >= output else 1


def calculate_term_grouping_distances(input: str) -> Tuple[float, float]:
    vars = re.findall(r"([a-zA-Z]\^?\d?)+", input)
    seen_pos: Dict[str, List[int]] = dict()
    for i, var in enumerate(vars):
        if var not in seen_pos:
            seen_pos[var] = []
        seen_pos[var].append(i)

    def get_var_signal(var, positions):
        out_signal = 0.0
        last_pos = -1
        for position in positions:
            if last_pos != -1:
                out_signal += position - last_pos
            last_pos = position
        return out_signal

    # seen_pos is a dict of positions that each variable is seen at
    # add up all the distances between the points in each variable
    signal = 0.0
    for key, value in seen_pos.items():
        signal += get_var_signal(key, value)

    # Scale the signal down to avoid it growing ever larger with more
    # complex inputs.
    return signal, signal / len(vars)


def calculate_grouping_control_signal(
    input: str, output: str, clip_at_zero: bool = False
) -> float:
    """Calculate grouping_control signals as the sum of all distances between
    all like terms. Gather all the terms in an expression and add an error value
    whenever a like term is separated by another term.

    Examples:
        "2x + 2x" = 0
        "2x + 4y + 2x" = 1
        "2x + 4y + 2x + 4y" = 2
        "2x + 2x + 4y + 4y" = 0
    """

    # We cheat the term grouping a bit by not parsing the expression
    # and finding the real set of terms. Instead we remove all the non-variables
    # and then count the distances from the resulting string.
    #
    # NOTE: this means that the signal is not correct when exponents or complex
    #       terms with multiple variables are in the expression. Perhaps it's a
    #       good improvement to make here.
    in_signal, in_signal_normalized = calculate_term_grouping_distances(input)
    out_signal, out_signal_normalized = calculate_term_grouping_distances(output)
    # The grouping distance stayed the same
    if in_signal == out_signal:
        return out_signal_normalized

    # It changed, no error
    if clip_at_zero is True:
        return 0.0

    # It changed, negative error based on magnitude of the change
    return -abs(in_signal_normalized - out_signal_normalized)


def calculate_group_prediction_signal(observation: MathyObservation):
    """Calculate the ratio of unique groups of like terms to all terms in
    the expression. This is useful to get a number that stays in range of 0-1
    so your loss does not run all over the place when you introduce variable
    length inputs. The division by the number of terms is important for making
    the problem difficult to predict, because it causes the value to change
    as the problem evolves over time. i.e. without the divide the model would
    easily be able to predict the number of fixed unique term groups in an expression
    because it wouldn't change for any of the transformations of that input in
    an episode.

    Examples:
        "2x + 4x" = 1 / 2
        "2x^2 + 3x" = 2 / 2
        "y + x + z" = 3 / 3
        "x + x + y + y + y" = 2 / 5

    """
    # We cheat the term grouping a bit by not parsing the expression
    # and finding the real set of terms. Instead we remove all the non-variables
    # and then count the distances from the resulting string.
    #
    # NOTE: this means that the signal is not correct when exponents or complex
    #       terms with multiple variables are in the expression. Perhaps it's a
    #       good improvement to make here.

    # "2x + 2x  + 4y + 4y" -> "xxyy"
    vars = re.sub(r"[^a-zA-Z]+", "", observation.input)
    unique_vars = set()
    for var in vars:
        unique_vars.add(var)
    return len(unique_vars) / len(vars)


def calculate_reward_prediction_signal(observation: MathyObservation):
    """reward_prediction signal is a single integer indicating one of three
    classes: POSITIVE, NEUTRAL, NEGATIVE based on the reward for
    entering the current state.
    """
    undiscounted = observation.discounted
    epsilon = 0.01
    neutral = 1 if undiscounted < epsilon and undiscounted > -epsilon else 0
    negative = 1 if not neutral and undiscounted < 0.0 else 0
    positive = 1 if not neutral and undiscounted > 0.0 else 0
    return [negative, neutral, positive]


def calculate_policy_target(observation: MathyObservation, soft: bool = False):
    policy = numpy.array(observation.policy[:], dtype="float32")
    # If we're using the hard targets, pass the policy distribution back
    # directly from the tree search. This may end up being the best way, but
    # I'm exploring "soft" targets after watching Jeff Dean talk at Stanford
    # for Karpathy's course.
    if soft is not True:
        return policy
    # The policy coming from the network will usually already include weight
    # for each active rule, but we want the model to be less sure of itself
    # with the targets so we adjust the policy weightings slightly.
    policy_mask = numpy.array(observation.features["policy_mask"][:], dtype="float32")
    # assert numpy.count_nonzero(policy_mask) == numpy.count_nonzero(policy)
    # The policy mask has 0.0 and 1.0 values. We scale the mask down so that
    # 1 becomes smaller, and then we elementwise add the policy and the mask to
    # increment each valid action by the scaled value
    policy_soft = policy + (policy_mask * 0.01)
    # Finally we re-normalize the values so that they sum to 1.0 like they
    # did before we incremented them.
    policy_soft /= numpy.sum(policy_soft)
    return policy_soft

