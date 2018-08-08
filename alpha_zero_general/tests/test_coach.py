from ..NeuralNet import NeuralNet
from ..Coach import Coach
from ..Game import Game
from ..EpisodeRunner import EpisodeRunner, RunnerConfig


class MockEpisodeRunner(EpisodeRunner):
    def get_game(self):
        return Game()

    def get_nnet(self, game, data=None):
        return NeuralNet(game)


def test_coach():
    config = RunnerConfig()
    runner = MockEpisodeRunner(config)
    c = Coach(runner)
    assert c is not None
