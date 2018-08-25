from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.math_players import RandomPlayer
from mathzero.math_neural_net import MathNeuralNet
import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = MathGame()

rp = RandomPlayer(g).play

n1 = MathNeuralNet(g)
n1.load_checkpoint("./training/latest/latest.pth.tar")
mcts1 = MCTS(g, n1, cpuct=1.0, num_mcts_sims=100, epsilon=0)
calvin = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# n2 = MathNeuralNet(g)
# n2.load_checkpoint("./training/08_11_18_1hr/latest.pth.tar")
# mcts2 = MCTS(g, n2, cpuct=1.0, num_mcts_sims=100, epsilon=0)
# algernon = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
algernon = rp

arena = Arena(calvin, algernon, g, display=display)
print(arena.playGames(20, verbose=True))
