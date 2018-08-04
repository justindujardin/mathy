from sys import stdin
from multiprocessing import Pool, Array, Process
import alpha_zero_general
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import executeEpisode
from alpha_zero_general.Game import Game
from mathzero.math_game import MathGame
from mathzero.math_neural_net import NNetWrapper


def execute_math_episode(key):
    game = MathGame()
    network = NNetWrapper(game)

    # game = alpha_zero_general.game
    # network = alpha_zero_general.network
    result = executeEpisode(game, network, 1, 25, 0.5, 1.0)
    return result


def initProcess(game, nnet):
    """This is to allow sharing data across process boundaries"""
    alpha_zero_general.network = nnet
    alpha_zero_general.game = game


if __name__ == "__main__":
    # game = MathGame()
    # nnet = NNetWrapper(game)
    # pool = Pool(initializer=initProcess, initargs=(game, nnet))
    pool = Pool()
    print(pool.map(execute_task, range(10)))

