from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.math_players import GreedyMathPlayer, RandomPlayer, PassPlayer
from mathzero.math_neural_net import MathNeuralNet
import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = MathGame()

# all players
rp = RandomPlayer(g).play
gp = GreedyMathPlayer(g).play
pp = PassPlayer(g).play

# nnet players
n1 = MathNeuralNet(g)
n1.load_checkpoint("./models/08_03_18/2/best.pth.tar")
mcts1 = MCTS(g, n1, cpuct=1.0, num_mcts_sims=100)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = MathNeuralNet(g)
n2.load_checkpoint('./training/temp/best.pth.tar')
mcts2 = MCTS(g, n2, cpuct=1.0, num_mcts_sims=100)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena(n1p, n2p, g, display=display)
print(arena.playGames(50, verbose=True))
