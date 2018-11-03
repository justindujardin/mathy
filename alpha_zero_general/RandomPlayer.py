import numpy as np
import random


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, env_state):
        a = np.random.randint(self.game.get_agent_actions_count())
        valids = self.game.getValidMoves(env_state, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_agent_actions_count())
        return a
