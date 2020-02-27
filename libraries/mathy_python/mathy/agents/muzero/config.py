from typing import Callable, Optional
from pydantic import BaseModel
from .types import Environment, KnownBounds


class MuZeroConfig(object):
    model_size: int
    # PyTorch models need padded batches of the max length a sequence might be.
    # Mathy expressions can occasionally go over 128 nodes, but I've never made
    # an expression more than 256. If we go over this limit, it might be time to
    # introduce the Reformer.
    max_sequence_length: int
    action_space_size: int
    max_moves: int
    discount: float
    dirichlet_alpha: float
    num_simulations: int
    batch_size: int
    td_steps: int
    num_actors: int
    lr_init: float
    lr_decay_steps: float
    visit_softmax_temperature_fn: Callable[[int, int], float]

    def __init__(
        self,
        model_width: int,
        action_space_size: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        num_actors: int,
        lr_init: float,
        lr_decay_steps: float,
        visit_softmax_temperature_fn,
        known_bounds: Optional[KnownBounds] = None,
    ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
        self.model_width = model_width

    def new_game(self, env: Environment):
        from .game import Game

        return Game(self.action_space_size, self.discount, env, self.model_width)
