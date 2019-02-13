import uuid
import json
from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.model.tensorflow_neural_net import MathNeuralNet
import numpy as np

game = MathGame(verbose=True, max_moves=100)
predictor = MathNeuralNet(game)
predictor.load_checkpoint("./training/web_2/latest.pth.tar")
mcts = MCTS(game, predictor, cpuct=1.0, num_mcts_sims=25, epsilon=0)
calvin = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
arena = Arena(calvin, game, display=display)
solved, failed, details = arena.playGames(10)
print(solved, failed)
meta_file = "visualization/arena_{}".format(uuid.uuid4().hex)
with open(meta_file, "w") as file:
    file.write(json.dumps(details, indent=2))
    print('wrote test results to "{}"'.format(meta_file))
