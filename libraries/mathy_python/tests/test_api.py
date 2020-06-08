import pytest
import tensorflow as tf

from mathy.agents.base_config import BaseConfig
from mathy.agents.policy_value_model import PolicyValueModel
from mathy.api import Mathy, MathyAPISwarmState


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


def test_api_mathy_constructor():
    # Defaults to swarm config
    assert isinstance(Mathy().state, MathyAPISwarmState)

    # Config must be a known pydantic config
    with pytest.raises(ValueError):
        Mathy(config={})  # type:ignore

    # Model must be PVM
    with pytest.raises(ValueError):
        Mathy(model=dict(), config=BaseConfig())  # type:ignore
    # Model Config must extend BaseConfig
    with pytest.raises(ValueError):
        Mathy(model=PolicyValueModel(), config=dict())  # type:ignore

    # Config by itself must be SwarmConfig()
    with pytest.raises(ValueError):
        Mathy(config=BaseConfig())
