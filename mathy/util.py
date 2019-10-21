import numpy as np
from tf_agents.trajectories import time_step


class GameRewards:
    """Game reward constant values"""

    LOSE = -1.0
    WIN = 1.0
    HELPFUL_MOVE = 0.01
    UNHELPFUL_MOVE = -0.01
    TIMESTEP = -0.01
    PREVIOUS_LOCATION = -0.05
    INVALID_ACTION = -0.3


def is_terminal_transition(transition: time_step.TimeStep):
    return bool(transition.step_type == time_step.StepType.LAST)


def is_win_reward(reward):
    return reward == GameRewards.WIN


def is_lose_reward(reward):
    return reward == GameRewards.LOSE


def discount(r, gamma=0.99):
    """Discount a list of float rewards to encourage rapid convergance.
    r: input array of floats
    gamma: a float value between 0 and 0.99"""
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    # reverse them to restore the correct order
    np.flip(discounted_r)
    return discounted_r


def normalize_rewards(r):
    """Normalize a set of rewards to values between -1 and 1"""
    d = 2 * (r - np.min(r)) / np.ptp(r) - 1
    return d


def pad_array(A, max_length, value=0, backwards=False, cleanup=False):
    """Pad a list to the given size with the given padding value

    If backwards=True the input will be reversed after padding, and
    the output will be reversed after padding, to correctly pad for
    LSTMs, e.g. "4x+2----" padded backwards would be "----2+x4"
    """
    if backwards:
        A.reverse()
    while len(A) < max_length:
        A.append(value)
    if backwards:
        A.reverse()
    if cleanup is True:
        A = np.array(A).tolist()
    return A

