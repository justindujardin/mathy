from typing import Any, List, Tuple

import pytest
from pydantic import BaseModel

from mathy import envs
from mathy.agents.policy_value_model import PolicyValueModel
from mathy.agents.zero.config import SelfPlayConfig
from mathy.agents.zero.practice_runner import PracticeRunner
from mathy.agents.zero.trainer import SelfPlayTrainer
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window


def test_mathy_zero_trainer_constructor():
    config = SelfPlayConfig()
    env: MathyEnv = envs.PolySimplify()
    model = PolicyValueModel(config, predictions=env.action_size)
    assert SelfPlayTrainer(config, model, env.action_size) is not None


def test_mathy_zero_practice_runner():
    with pytest.raises(ValueError):
        # Must be SelfPlayConfig or a subclass
        PracticeRunner(BaseModel())

    runner = PracticeRunner(SelfPlayConfig())
    with pytest.raises(NotImplementedError):
        runner.get_env()
    with pytest.raises(NotImplementedError):
        runner.get_model(None)


def test_mathy_zero_practice_runner_execute_episode():
    runner = PracticeRunner(SelfPlayConfig())

    class FakeEnv:
        def __init__(self):
            self.state = None

        def reset(self):
            pass

    with pytest.raises(ValueError):
        runner.execute_episode(1, None, None, "", False)
    with pytest.raises(ValueError):
        runner.execute_episode(1, FakeEnv(), None, "", False)
    with pytest.raises(ValueError):
        runner.execute_episode(1, FakeEnv(), FakeEnv(), "", False)
