from typing import List, Tuple

import pytest

from ..mathy import envs
from ..mathy.agents.policy_value_model import PolicyValueModel
from ..mathy.agents.zero.config import SelfPlayConfig
from ..mathy.agents.zero.trainer import SelfPlayTrainer
from ..mathy.env import MathyEnv
from ..mathy.state import MathyObservation, observations_to_window


def test_mathy_zero_trainer_constructor():
    config = SelfPlayConfig(use_grouping_control=True)
    env: MathyEnv = envs.PolySimplify()
    model = PolicyValueModel(config, predictions=env.action_size)
    with pytest.raises(NotImplementedError):
        SelfPlayTrainer(config, model, env.action_size)
