from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.math_players import GreedyMathPlayer, RandomPlayer
from mathzero.math_neural_net import NNetWrapper as NNet
import numpy as np
from alpha_zero_general.utils import dotdict

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = MathGame()

# all players
rp = RandomPlayer(g).play
gp = GreedyMathPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint("./training/gpu_2/temp/best.pth.tar")
mcts1 = MCTS(g, n1, cpuct=1.0, num_mcts_sims=50)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./training/temp/checkpoint_1.pth.tar')
mcts2 = MCTS(g, n2, cpuct=1.0, num_mcts_sims=50)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


arena = Arena(n1p, n2p, g, display=display)
print(arena.playGames(10, verbose=True))
