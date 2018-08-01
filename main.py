from alpha_zero_general.Coach import Coach
from mathzero.math_game import MathGame
from mathzero.math_neural_net import NNetWrapper as nn
from mathzero.core.expressions import ConstantExpression
from mathzero.core.parser import ExpressionParser

eps = 250
temp = int(eps * 0.5)
arena = min(int(eps * 0.3), 30)

args = {
    "training_iterations": 1000,
    "self_play_iterations": eps,
    "temperature_threshold": temp,
    "model_win_loss_ratio": 0.6,
    "max_training_examples": 200000,
    "num_mcts_sims": 20,
    "model_arena_iterations": arena,
    "cpuct": 1,
    "checkpoint": "./training/temp/",
    "best_model_name": "best",
    "save_examples_from_last_n_iterations": 20,
}


if __name__ == "__main__":
    # parser = ExpressionParser()
    # expression = parser.parse('7 + x + 2 - 2x')
    # expression = parser.parse("4 + x + 3")
    # print("Expression \"{}\" evaluates to: {}".format(expression, expression.evaluate()))
    # expression = parser.parse('1100 - 100 + 300 + 37')
    # expression = parser.parse('(7 - (5 - 3)) * (32 - 7)')
    g = MathGame()
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
