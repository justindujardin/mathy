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
from ..mathy_model import MathyModel

Environment = PolySimplify

MAXIMUM_FLOAT_VALUE = float("inf")

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class Player(MathyAgentState):
    def __init__(self):
        super(Player, self).__init__(10, "4x+2x", -1)


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    maximum: float
    minimum: float

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroConfig(object):
    def __init__(
        self,
        model_width: int,  # TODO: remove this? hack to pad sequences when making observations
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

    def new_game(self, env: Environment = None):
        return Game(self.action_space_size, self.discount, env, self.model_width)


@dataclass
class Action(object):
    index: int

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


@dataclass
class Node(object):
    prior: float
    visit_count: int = 0
    to_play: int = 0
    value_sum: float = 0
    children: Dict[str, "Node"] = field(default_factory=dict)
    hidden_state: Optional[List[float]] = None
    reward: float = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory(object):
    """MCTS history container used inside the search.

  Only used to keep track of the actions executed.
  """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> MathyAgentState:
        return MathyAgentState(-1, "4x+2x", -1)


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(MathyModel):
    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        out = self(image)
        return out

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


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


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: MuZeroConfig, root: Node, action_history: ActionHistory, network: Network
):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        assert len(search_path) > 2, "search terminated without a long enough path"

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.hidden_state, history.last_action()
        )
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats
) -> float:
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
    node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput
):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
