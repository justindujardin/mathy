import collections
from dataclasses import dataclass, field
from typing import Dict, Generic, List, NamedTuple, Optional, TypeVar

import numpy

from ...envs.poly_simplify import PolySimplify
from ...state import (
    MathyAgentState,
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    MathyInputsType,
)
from ...state import observations_to_window
from .network import Network, NetworkOutput
from .types import (
    Environment,
    ActionHistory,
    Action,
    Node,
    Player,
    MinMaxStats,
)
from .config import MuZeroConfig


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(
        self,
        action_space_size: int,
        discount: float,
        env: Environment,
        model_width: int,
    ):
        self.environment = env
        self.state, self.problem = self.environment.get_initial_state()
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False
        self.model_width = model_width

    def terminal(self) -> bool:
        return self.done is True

    def legal_actions(self, current_observation) -> List[Action]:
        # Game specific calculation of legal actions.
        mask = self.environment.get_valid_moves(self.state)
        return [Action(index) for index, valid in enumerate(mask) if valid == 1]

    def apply(self, action: Action):
        observation, reward, done, info = self.environment.step(action)
        self.done = done
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def step(self, action):
        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        observation = self._observe(self.state)
        info = {"transition": transition}
        done = is_terminal_transition(transition)
        self.last_action = action
        self.last_change = change
        self.last_reward = round(float(transition.reward), 4)
        return observation, transition.reward, done, info

    def make_image(self, state_index: int) -> MathyInputsType:
        obs = self.environment.state_to_observation(self.state, rnn_size=1)
        return observations_to_window([obs], self.model_width).to_inputs()

    def make_target(
        self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player
    ):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += (
                    reward * self.discount ** i
                )  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append(
                    (
                        value,
                        self.rewards[current_index],
                        self.child_visits[current_index],
                    )
                )
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network, env: Environment) -> Game:
    game = config.new_game(env)

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(
            root,
            game.to_play(),
            game.legal_actions(current_observation),
            network.initial_inference(current_observation),
        )
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game
