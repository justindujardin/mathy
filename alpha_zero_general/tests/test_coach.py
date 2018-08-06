from ..NeuralNet import NeuralNet
from ..Coach import Coach
from ..Game import Game


def test_coach():
    g = Game()
    nnet = NeuralNet(g)
    c = Coach(g, nnet)
    assert c is not None
