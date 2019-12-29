from typing import Tuple
from ..mathy.agents.episode_memory import EpisodeMemory
from ..mathy.envs.poly_simplify import PolySimplify
from numpy.testing import assert_array_almost_equal


def get_memory(
    number_observations: int, rnn_size: int = 4
) -> Tuple[EpisodeMemory, PolySimplify]:
    """Verify that RNN history is the average of the RNN states in memory"""
    memory = EpisodeMemory()
    env = PolySimplify()
    state, problem = env.get_initial_state()
    for i in range(number_observations):
        memory.store(
            observation=env.state_to_observation(state, rnn_size=rnn_size),
            action=0,
            reward=0.0,
            grouping_change=0.0,
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
    obs = env.state_to_observation(env.get_initial_state()[0], rnn_size=4)
    # Returns only the last three
    assert len(memory.to_window_observation(obs, window_size=3).nodes) == 3
    # Only returns as many observations as it has (plus the one provided)
    assert len(memory.to_window_observation(obs, window_size=10).nodes) == 5


def test_episode_memory_rnn_history():
    """Verify that RNN history is the average of the RNN states in memory"""

    memory = EpisodeMemory()
    env = PolySimplify()
    state, problem = env.get_initial_state()
    # Two RNN states
    rnn_states = [
        [[0.0, 1.0, 0.8], [1.2, 0.2, 1.8]],
        [[0.5, 1.0, 0.4], [0.0, 2.2, 0.4]],
    ]
    # The expected element-wise average of those states
    rnn_assert = [[0.25, 1.0, 0.6], [0.6, 1.2, 1.1]]

    # Insert the RNN states into memory
    for r_state in rnn_states:
        memory.store(
            observation=env.state_to_observation(
                state, rnn_state_h=r_state[0], rnn_state_c=r_state[1]
            ),
            action=0.0,
            reward=0.0,
            grouping_change=0.0,
            value=0.0,
        )
    # Fetch and check the weighted RNN history value
    weighted_rnn = memory.rnn_weighted_history(len(rnn_assert[0]))
    assert_array_almost_equal(weighted_rnn, rnn_assert)
