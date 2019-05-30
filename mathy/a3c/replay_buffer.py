from typing import List


class ReplayBuffer(object):
    states: List[str] = []
    actions: List[int] = []
    rewards: List[float] = []

    def __init__(self):
        self.states = [True]
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

