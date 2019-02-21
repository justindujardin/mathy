import uuid
import json
from alpha_zero_general.ExaminationRunner import ExaminationRunner
from alpha_zero_general.MCTS import MCTS
from mathzero.math_game import MathGame, display
from mathzero.model.math_model import MathModel
import numpy as np

game = MathGame(verbose=True, max_moves=25)
predictor = MathModel(game, "./training/embedding_3")
predictor.start()
mcts = MCTS(game, predictor, cpuct=1.0, num_mcts_sims=50, epsilon=0)
calvin = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
arena = ExaminationRunner(calvin, game, display=display)
solved, failed, details = arena.playGames(100)
print(solved, failed)
meta_file = "visualization/arena_{}".format(uuid.uuid4().hex)
with open(meta_file, "w") as file:
    file.write(json.dumps(details, indent=2))
    print('wrote test results to "{}"'.format(meta_file))
predictor.stop()
