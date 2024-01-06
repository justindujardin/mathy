import pytest
from mathy.api import Mathy, MathyAPISwarmState


def test_api_mathy_constructor():
    # Defaults to swarm config
    assert isinstance(Mathy().state, MathyAPISwarmState)

    # Config must be a known pydantic config
    with pytest.raises(ValueError):
        Mathy(config={})  # type:ignore


@pytest.mark.parametrize("single_process", [True, False])
def test_api_mathy_solver(single_process: bool):
    from mathy.api import Mathy
    from mathy.solver import SwarmConfig

    mt = Mathy(
        config=SwarmConfig(use_mp=not single_process, n_walkers=512, verbose=True)
    )
    mt.simplify(problem="4z+3z", max_steps=20)
