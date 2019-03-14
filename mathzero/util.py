import numpy

# From TZ: "my rule of thumb is win/loss = +/-1, and everything else is determined in orders of magnitude of importance
# so for instance, my timestep penalty might be -0.01, picking up a gem or something might be +0.1"
REWARD_LOSE = -1
REWARD_WIN = 1
REWARD_TIMESTEP = -0.01
REWARD_NEW_LOCATION = 0.001
REWARD_PREVIOUS_LOCATION = -0.02
REWARD_INVALID_ACTION = -0.02


def is_terminal_reward(reward):
    return reward == REWARD_WIN or reward == REWARD_LOSE


def is_win_reward(reward):
    return reward == REWARD_WIN


def is_lose_reward(reward):
    return reward == REWARD_LOSE


def discount(r, gamma=0.99):
    """Discount a list of float rewards to encourage rapid convergance"""
    discounted_r = numpy.zeros_like(r, dtype=numpy.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def normalize_rewards(r):
    """Normalize a set of rewards to values between -1 and 1"""
    d = 2 * (r - numpy.min(r)) / numpy.ptp(r) - 1
    return d

