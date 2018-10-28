from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.math_players import RandomPlayer
from mathzero.model.tensorflow_neural_net import MathNeuralNet
import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = MathGame(verbose=True)
predictor = MathNeuralNet(game)
predictor.load_checkpoint("./training/latest_harder/latest.pth.tar")
mcts = MCTS(game, predictor, cpuct=1.0, num_mcts_sims=200, epsilon=0)
calvin = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
arena = Arena(calvin, game, display=display)
print(arena.playGames(20))
