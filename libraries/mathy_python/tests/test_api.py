import pytest
import tensorflow as tf

from mathy.agent.config import AgentConfig
from mathy.agent.model import AgentModel, build_agent_model
from mathy.api import Mathy, MathyAPISwarmState


def test_mathy_with_model_and_config():
    config = AgentConfig()
    model: AgentModel = build_agent_model(config, predictions=2)
    mt = Mathy(model=model, config=config)


def test_api_mathy_constructor():
    # Defaults to swarm config
    assert isinstance(Mathy().state, MathyAPISwarmState)

    # Config must be a known pydantic config
    with pytest.raises(ValueError):
        Mathy(config={})  # type:ignore

    # Model must be a keras model
    with pytest.raises(ValueError):
        Mathy(model=dict(), config=AgentConfig())  # type:ignore
    # Model Config must extend AgentConfig
    with pytest.raises(ValueError):
        Mathy(
            model=build_agent_model(AgentConfig()), config=dict(),  # type:ignore
        )

    # Config by itself must be SwarmConfig()
    with pytest.raises(ValueError):
        Mathy(config=AgentConfig())
