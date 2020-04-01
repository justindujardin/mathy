from typing import Tuple
from mathy.agents.episode_memory import EpisodeMemory
from mathy.envs.poly_simplify import PolySimplify
from numpy.testing import assert_array_almost_equal


def get_memory(number_observations: int) -> Tuple[EpisodeMemory, PolySimplify]:
    memory = EpisodeMemory()
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
