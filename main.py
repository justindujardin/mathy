from Coach import Coach
from mathzero.MathGame import MathGame
from mathzero.tensorflow.NNet import NNetWrapper as nn
from mathzero.math.expressions import ConstantExpression
from mathzero.math.parser import ExpressionParser
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    parser = ExpressionParser()
    # expression = parser.parse('7 + x + 2')
    # expression = parser.parse('7 + 5 + 2 * 2 + 14 + 6')
    expression = parser.parse('7 + 5 * 2')
    g = MathGame(expression)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
