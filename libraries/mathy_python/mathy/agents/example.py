from typing import Optional

from ..envs import PolySimplify
from ..state import MathyWindowObservation, observations_to_window
from .base_config import BaseConfig


def example() -> MathyWindowObservation:
    """Helper to return a random Window observation that can be 
    passed forward through a Mathy model. """
    env = PolySimplify()
    state = env.get_initial_state()[0]
    observation = env.state_to_observation(state, rnn_size=BaseConfig().lstm_units)
    return observations_to_window([observation])
