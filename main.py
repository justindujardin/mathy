import sys
from mathzero.training.practice_session import PracticeSession
from mathzero.math_game import MathGame
from mathzero.model.math_model import MathModel
from mathzero.core.expressions import ConstantExpression
from mathzero.core.parser import ExpressionParser
from mathzero.training.practice_runner import (
    PracticeRunner,
    ParallelPracticeRunner,
    RunnerConfig,
)

eps = 100

args = {"self_play_iterations": eps, "max_training_examples": 200000}

# NOTE: For a new model bootstrap, it won't use examples file if there's not a checkpoint found.
# TODO: Fix this :point_up:

# Single-process implementation for debugging and development
dev_mode = False

BaseEpisodeRunner = PracticeRunner if dev_mode else ParallelPracticeRunner

# model_dir = "./training/embedding_1/"
model_dir = "/mnt/gcs/mzc/embedding_5/"


class MathEpisodeRunner(BaseEpisodeRunner):
    def get_game(self):
        return MathGame(verbose=dev_mode)

    def get_predictor(self, game, all_memory=False):
        return MathModel(game, model_dir, all_memory)


if __name__ == "__main__":
    config = RunnerConfig(
        model_dir=model_dir,
        num_mcts_sims=(100 if dev_mode else 1500),
        temperature_threshold=round(MathGame.max_moves_easy * 0.7),
        cpuct=1.0,
    )
    runner = MathEpisodeRunner(config)
    c = PracticeSession(runner, args)
    if not c.has_examples:
        prompt = "No existing examples found. Session will build new training examples. Continue? [y|N]"
        r = input(prompt)
        if r != "y":
            sys.exit()
        print(
            "No existing checkpoint found, starting with a fresh model and self-play..."
        )

    c.learn()
