from alpha_zero_general.EpisodeRunner import (
    EpisodeRunner,
    ParallelEpisodeRunner,
    RunnerConfig,
)
import sys
from alpha_zero_general.Coach import Coach
from mathzero.math_game import MathGame
from mathzero.math_neural_net import MathNeuralNet
from mathzero.core.expressions import ConstantExpression
from mathzero.core.parser import ExpressionParser

eps = 100
temp = int(eps * 0.5)

args = {
    "training_iterations": 1000,
    "self_play_iterations": eps,
    "max_training_examples": 200000,
    "checkpoint": "./training/temp/",
    "best_model_name": "latest",
}

# Single-process implementation for debugging and development
dev_mode = False

BaseEpisodeRunner = EpisodeRunner if dev_mode else ParallelEpisodeRunner


class MathEpisodeRunner(BaseEpisodeRunner):
    def get_game(self):
        return MathGame()

    def get_nnet(self, game, all_memory=False):
        return MathNeuralNet(game, all_memory)


if __name__ == "__main__":
    config = RunnerConfig(num_mcts_sims=50, temperature_threshold=temp, cpuct=1.0)
    runner = MathEpisodeRunner(config)
    c = Coach(runner, args)
    if not c.has_examples:
        prompt = "No existing examples found. Session will build new training examples. Continue? [y|N]"
        r = input(prompt)
        if r != "y":
            sys.exit()
        print(
            "No existing checkpoint found, starting with a fresh model and self-play..."
        )

    # Run the self-play/train/compare loop for (n) iterations
    c.learn()
