from alpha_zero_general.Coach import Coach
from alpha_zero_general.utils import dotdict
from mathzero.math_game import MathGame
from mathzero.math_neural_net import NNetWrapper as nn
from mathzero.math.expressions import ConstantExpression
from mathzero.math.parser import ExpressionParser

eps = 10
temp = int(eps * 0.5)
arena = int(eps * 0.6)

args = dotdict(
    {
        "numIters": 1000,
        "numEps": eps,
        "tempThreshold": temp,
        "updateThreshold": 0.6,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 15,
        "arenaCompare": arena,
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./pretrained_models/temp/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)

if __name__ == "__main__":
    # parser = ExpressionParser()
    # expression = parser.parse('7 + x + 2 - 2x')
    # expression = parser.parse("4 + x + 3")
    # print("Expression \"{}\" evaluates to: {}".format(expression, expression.evaluate()))
    # expression = parser.parse('1100 - 100 + 300 + 37')
    # expression = parser.parse('(7 - (5 - 3)) * (32 - 7)')
    g = MathGame()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
