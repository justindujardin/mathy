import pytest
import tensorflow as tf

from mathy import envs
from mathy.agents.base_config import BaseConfig
from mathy.agents.fragile import SwarmConfig
from mathy.agents.policy_value_model import PolicyValueModel
from mathy.api import Mathy, MathyAPIModelState, MathyAPISwarmState
from mathy.env import MathyEnv
from mathy.state import MathyObservation, observations_to_window


def test_mathy_policy_value_subclass_error():
    """Do not allowing entirely custom models. They must subclass PolicyValueModel"""
    # NOTE: if this test is breaking your model, let's talk about it on Github
    model = tf.keras.Model()
    with pytest.raises(ValueError):
        mt = Mathy(model=model, config=BaseConfig())


def test_mathy_with_model_and_config():
    config = BaseConfig()
    model = PolicyValueModel(config, predictions=2)
    mt = Mathy(model=model, config=config)


def test_api_mathy_simplify():
    mt: Mathy = Mathy(config=SwarmConfig())
    mt.simplify(problem="2x+4x", max_steps=4)


def test_api_mathy_constructor():
    # Defaults to swarm config
    assert isinstance(Mathy().state, MathyAPISwarmState)

    # Config must be a known pydantic config
    with pytest.raises(ValueError):
        Mathy(config={})

    # Model must be PVM
    with pytest.raises(ValueError):
        Mathy(model=dict(), config=BaseConfig())
    # Model Config must extend BaseConfig
    with pytest.raises(ValueError):
        Mathy(model=PolicyValueModel(), config=dict())

    # Config by itself must be SwarmConfig()
    with pytest.raises(ValueError):
        Mathy(config=BaseConfig())
