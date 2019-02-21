import numpy

LOSE_REWARD = -100000
WIN_REWARD = 100000


def is_terminal_reward(reward):
    return reward >= WIN_REWARD or reward <= LOSE_REWARD


def is_win_reward(reward):
    return reward >= WIN_REWARD


def is_lose_reward(reward):
    return reward <= LOSE_REWARD


def discount_rewards(r, gamma=0.99):
    """Discount a list of float rewards to encourage rapid convergance"""
    discounted_r = numpy.zeros_like(r, dtype=numpy.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
