from mathy.mathy_env import MathyEnv
from mathy.agent.controller import MathModel
from mathy.agent.training.practice_runner import PracticeRunner, RunnerConfig
from mathy.agent.training.practice_session import PracticeSession


class MockEpisodeRunner(PracticeRunner):
    def get_game(self):
        return MathyEnv()

    def get_predictor(self, game, data=None):
        return MathModel(game, "/dev/null")


def test_coach():
    config = RunnerConfig()
    runner = MockEpisodeRunner(config)
    c = PracticeSession(runner)
    assert c is not None
