from typing import List, Tuple

import pytest

from mathy.envs.poly_simplify import PolySimplify
from mathy.state import MathyEnvState


def test_mathy_features_from_state():
    state = MathyEnvState(problem="4x+2x")
    assert state.to_observation([]) is not None


def test_mathy_features_hierarchy():
    """Verify that the observation generated encodes hierarchy properly
    so the model can determine the precise nodes to act on"""

    diff_pairs: List[Tuple[str, str]] = [
        ("4x + (3u + 7x + 3u) + 4u", "4x + 3u + 7x + 3u + 4u"),
        ("7c * 5", "7 * (c * 5)"),
        ("5v + 20b + (10v + 7b)", "5v + 20b + 10v + 7b"),
        ("5s + 60 + 12s + s^2", "5s + 60 + (12s + s^2)"),
    ]
    env = PolySimplify()

    for one, two in diff_pairs:
        state_one = MathyEnvState(problem=one)
        obs_one = state_one.to_observation(env.get_valid_moves(state_one))

        state_two = MathyEnvState(problem=two)
        obs_two = state_two.to_observation(env.get_valid_moves(state_two))

        assert obs_one.nodes != obs_two.nodes
