import pytest
from mathy.api import Mathy, MathyAPISwarmState


def test_api_mathy_constructor():
    # Defaults to swarm config
    assert isinstance(Mathy().state, MathyAPISwarmState)

    # Config must be a known pydantic config
    with pytest.raises(ValueError):
        Mathy(config={})  # type:ignore
