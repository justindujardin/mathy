from sys import stdin
from multiprocessing import Pool, Array, Process, Queue, cpu_count
import alpha_zero_general
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.Coach import executeEpisode
from alpha_zero_general.Game import Game
from mathzero.math_game import MathGame
from mathzero.math_neural_net import NNetWrapper
from time import sleep


def worker(work_queue, result_queue):
    """Pull items out of the work queue and execute episodes until there are no items left"""
    results = []
    while work_queue.empty() == False:
        episode = work_queue.get()
        results.append(execute_math_episode(episode))
    result_queue.put(results)
    return 0


def execute_math_episode(episode_number):
    game = MathGame()
    network = NNetWrapper(game)
    network.load_checkpoint("models/08_03_18/2/best.pth.tar")
    print("Starting episode: {}".format(episode_number))
    # game = alpha_zero_general.game
    # network = alpha_zero_general.network
    result = executeEpisode(game, network, 1, 25, 0.5, 1.0)
    return result


def initProcess(game, nnet):
    """This is to allow sharing data across process boundaries"""
    alpha_zero_general.network = nnet
    alpha_zero_general.game = game

def parallel_self_play_runner(episodes):
    # Fill a work queue with episodes to be executed.
    # NOTE: This is a bit overkill to pass episode numbers, but if you needed
    #       to pass more complex arguments because of some statefullness this
    #       would be useful.
    episodes = 4
    work_queue = Queue()
    result_queue = Queue()
    for i in range(episodes):
        work_queue.put(i)
    # pool = Pool(initializer=initProcess, initargs=(game, nnet))
    # game = MathGame()
    # nnet = NNetWrapper(game)
    # pool = Pool()
    processes = [
        Process(target=worker, args=(work_queue, result_queue))
        for i in range(cpu_count())
    ]
    for proc in processes:
        proc.start()

    # Gather the outputs
    results = []
    while len(results) != episodes:
        results.append(result_queue.get())

    # Wait for the workers to exit completely
    for proc in processes:
        proc.join()

    print(len(results))



if __name__ == "__main__":
    episodes = 4
    parallel_self_play_runner(episodes)
