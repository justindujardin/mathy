from typing import Tuple

import pytest
from numpy.testing import assert_array_almost_equal

from mathy.agent.episode_memory import EpisodeMemory
from mathy.envs.poly_simplify import PolySimplify


def get_memory(number_observations: int) -> Tuple[EpisodeMemory, PolySimplify]:
    memory = EpisodeMemory(128)
    env = PolySimplify()
    state, problem = env.get_initial_state()
    for i in range(number_observations):
        memory.store(
            observation=env.state_to_observation(state),
            action=0,
            reward=0.0,
            value=0.0,
        )
    return memory, env


def test_episode_memory_to_episode_window():
    """verify that to_episode_window returns a window of the entire episode"""
    memory, env = get_memory(12)

    assert len(memory.observations) == 12

    assert len(memory.to_episode_window().nodes) == 12


def test_episode_memory_to_window_observation():
    """Verify that to_window_observation only returns a subset of the memory"""
    memory, env = get_memory(4)
    assert len(memory.observations) == 4
    obs = env.state_to_observation(env.get_initial_state()[0])
    # Returns only the last three
    assert len(memory.to_window_observation(obs, window_size=3).nodes) == 3
    # Only returns as many observations as it has (plus the one provided)
    assert len(memory.to_window_observation(obs, window_size=10).nodes) == 5


def test_episode_memory_to_window_observations():
    """Split the memory into overlapping windows of the given size"""
    num_observations = 4
    memory, env = get_memory(num_observations)
    assert len(memory.observations) == num_observations
    # With 4 obs, there are only two full windows given stride = 1
    windows = memory.to_window_observations(window=3, stride=1, only_full_windows=True)
    assert len(windows) == 2

    # If you don't care about full windows, there are 4
    windows = memory.to_window_observations(window=3, stride=1, only_full_windows=False)
    assert len(windows) == 4

    # raises if given zip_with that doesn't have the same length as observations
    with pytest.raises(AssertionError):
        memory.to_window_observations(
            window=3, stride=1, only_full_windows=True, zip_with=[1]
        )

    # Can zip windows with a given list of other objects
    windows_and_values = memory.to_window_observations(
        window=3, stride=1, only_full_windows=True, zip_with=[1] * num_observations
    )
    for window, values in windows_and_values:
        assert len(window.nodes) == 3
        assert len(values) == 3

    # Can include other keys in output
    windows_and_actions_and_rewards_and_values = memory.to_window_observations(
        window=3,
        stride=1,
        only_full_windows=True,
        zip_with=[1] * num_observations,
        other_keys=["actions", "rewards"],
    )
    for window, actions, rewards, values in windows_and_actions_and_rewards_and_values:
        assert len(window.nodes) == 3
        assert len(values) == 3

