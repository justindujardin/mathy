import numpy
from tf_agents.trajectories import time_step


class GameRewards:
    """Game reward constant values"""

    LOSE = -1.0
    WIN = 1.0
    HELPFUL_MOVE = 0.01
    TIMESTEP = -0.01
    PREVIOUS_LOCATION = -0.2
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
    discounted_r = numpy.zeros_like(r, dtype=numpy.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    # reverse them to restore the correct order
    numpy.flip(discounted_r)
    return discounted_r


def normalize_rewards(r):
    """Normalize a set of rewards to values between -1 and 1"""
    d = 2 * (r - numpy.min(r)) / numpy.ptp(r) - 1
    return d

