import pytest
import tensorflow as tf

from ..mathy import envs
from ..mathy.agents.base_config import BaseConfig
from ..mathy.agents.policy_value_model import PolicyValueModel
from ..mathy.env import MathyEnv
from ..mathy.mathy import Mathy
from ..mathy.state import MathyObservation, observations_to_window


def test_mathy_requires_model_and_config_or_path():
    with pytest.raises(ValueError):
        mt = Mathy()


def test_mathy_policy_value_subclass_error():
    """Do not allowing entirely custom models. They must subclass PolicyValueModel"""
    # NOTE: if this test is breaking your model, let's talk about it on Github
    model = tf.keras.Model()
    with pytest.raises(ValueError):
        mt = Mathy(model=model)


def test_mathy_with_model_and_config():
    config = BaseConfig()
    model = PolicyValueModel(config, predictions=2)
    mt = Mathy(model=model, config=config)

