from sys import stdin
from multiprocessing import Pool, Array, Process
import alpha_zero_general
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import Coach, executeEpisode
from alpha_zero_general.Game import Game

args = {}


def execute_task(key):
    # print("Process loaded with arg: {}".format(key))
    game = alpha_zero_general.game
    network = alpha_zero_general.network
    # print("Shared game: {}".format(game))
    # print("Shared network: {}".format(network))
    c = Coach(game, network)
    # print("Coach is: {}".format(c))
    result = executeEpisode(game, network, 1, 25, 0.5, 1.0)
    return result


def initProcess(game, nnet):
    """This is to allow sharing data across process boundaries"""
    alpha_zero_general.network = nnet
    alpha_zero_general.game = game


if __name__ == "__main__":
    game = Game()
    nnet = NeuralNet(game)
    pool = Pool(initializer=initProcess, initargs=(game, nnet))
    print(pool.map(execute_task, ["a", "b", "s", "d"]))

