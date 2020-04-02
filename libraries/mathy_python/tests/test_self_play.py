import pytest

from mathy.agents.zero import SelfPlayConfig, self_play_runner


def test_self_play_runner_errors():
    """Throws error for bad profiling configuration of zero agent.
    
    This is only because it wasn't clear how to profile multiple processes that
    stop/start overtime during training/evaluation loops"""
    with pytest.raises(NotImplementedError):
        self_play_runner(SelfPlayConfig(profile=True, num_workers=2))
