from typing import Any, List, Tuple

import pytest
from pydantic import BaseModel

from ..mathy import envs
from ..mathy.agents.zero.config import SelfPlayConfig
from ..mathy.env import MathyEnv
from ..mathy.state import MathyObservation, observations_to_window
from ..mathy.agents.zero import SelfPlayConfig
from ..mathy.agents.muzero.config import MuZeroConfig
from ..mathy.agents.muzero.muzero import (
    ActionHistory,
    run_mcts,
    play_game,
    Node,
    Network,
    NetworkOutput,
)


def test_mathy_muzero_basic():
    config = SelfPlayConfig()
    env: MathyEnv = envs.PolySimplify()
    max_sequence_length = 256
    action_space_size = env.action_size * max_sequence_length
    max_moves = env.max_moves
    dirichlet_alpha = 0.03
    lr_init = 0.1

    def visit_softmax_temperature(num_moves: int, training_steps: int) -> float:
        if num_moves < 10:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    cfg = MuZeroConfig(
        model_width=config.max_sequence_length,
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=4,  # 3000
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
    )
    network = Network(config, actions_per_node=env.action_size)
    play_game(cfg, network, env)
