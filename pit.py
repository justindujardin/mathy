from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.math_players import GreedyMathPlayer, RandomPlayer, PassPlayer
from mathzero.math_neural_net import NNetWrapper as NNet
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
n1 = NNet(g)
n1.load_checkpoint("./training/gpu_4/checkpoint_19.pth.tar")
mcts1 = MCTS(g, n1, cpuct=1.0, num_mcts_sims=15)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./training/gpu_3/checkpoint_9.pth.tar')
mcts2 = MCTS(g, n2, cpuct=1.0, num_mcts_sims=15)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


arena = Arena(n1p, pp, g, display=display)
print(arena.playGames(10, verbose=True))
