from itertools import groupby
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from tf_agents.trajectories import time_step

from .core.expressions import STOP, MathExpression
from .core.parser import ExpressionParser
from .helpers import TermEx, compare_expression_string_values, get_term_ex
from .rules import (
    AssociativeSwapRule,
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ExpressionChangeRule,
    VariableMultiplyRule,
)
from .state import (
    MathyEnvState,
    MathyEnvTimeStep,
    MathyObservation,
    RNNStatesFloatList,
    rnn_placeholder_state,
)
from .types import MathyEnvProblem, MathyEnvProblemArgs
from .util import GameRewards


def mathy_core_rules(preferred_term_commute=False) -> List[BaseRule]:
    """Return the mathy core agent actions"""
    return [
        ConstantsSimplifyRule(),
        CommutativeSwapRule(preferred=preferred_term_commute),
        DistributiveMultiplyRule(),
        DistributiveFactorOutRule(),
        AssociativeSwapRule(),
        VariableMultiplyRule(),
    ]


class MathyEnv:
    """Implement a math solving game where a player wins by executing the
    right sequence of actions to reduce a math expression to an agreeable
    basic representation in as few moves as possible."""

    rules: List[BaseRule]
    rewarding_actions: List[Type[BaseRule]]
    max_moves: int
    verbose: bool
    reward_discount: float
    parser: ExpressionParser
    valid_actions_mask_cache: Dict[str, List[int]]
    valid_rules_cache: Dict[str, List[int]]

    def __init__(
        self,
        rules=None,
        rewarding_actions=None,
        max_moves=20,
        verbose=False,
        reward_discount=0.99,
    ):
        self.discount = reward_discount
        self.verbose = verbose
        self.max_moves = max_moves
        self.parser = ExpressionParser()
        self.rules = rules
        if self.rules is None:
            self.rules = mathy_core_rules()
        self.rewarding_actions = rewarding_actions
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()

    @property
    def action_size(self) -> int:
        """Return the number of available actions"""
        return len(self.rules)

    def finalize_state(self, state: MathyEnvState):
        """Perform final checks on a problem state, to ensure the episode yielded results
        that are uncorrupted by transformation errors. """
        from_timestep: MathyEnvTimeStep = state.agent.history[0]
        to_timestep: MathyEnvTimeStep = state.agent.history[-1]
        compare_expression_string_values(str(from_timestep.raw), str(to_timestep.raw))

    def get_env_namespace(self) -> str:
        """Return a unique dot namespaced string representing the current
        environment. e.g. mycompany.envs.differentiate"""
        raise NotImplementedError("subclass must implement this")

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        """Get the list of rewarding action types. When these actions
        are selected, the agent gets a positive reward. """
        # NOTE: by default we give a positive reward for most actions taken. Reward
        #       values are only applied AFTER penalties, so things like reentrant
        #       states become negative reward even if their action is otherwise
        #       rewarding.
        return [
            ConstantsSimplifyRule,
            DistributiveMultiplyRule,
            DistributiveFactorOutRule,
            VariableMultiplyRule,
        ]

    def get_penalizing_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        """Get the list of penalizing action types. When these actions
        are selected, the agent gets a negative reward."""
        return []

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        """Return the environment specific maximum move count for a given prolem."""
        return problem.complexity * 3

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
    ) -> Optional[time_step.TimeStep]:
        """Provide environment-specific transitions per timestep."""
        return None

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        """Return a problem for the environment given a set of parameters
        to control problem generation.

        This is implemented per environment such that each environment can
        generate its own dataset with no required configuration."""
        raise NotImplementedError("This must be implemented in a subclass")

    def state_to_observation(
        self,
        state: MathyEnvState,
        rnn_size: Optional[int] = None,
        rnn_state: Optional[RNNStatesFloatList] = None,
        rnn_history: Optional[RNNStatesFloatList] = None,
    ) -> MathyObservation:
        """Convert an environment state into an observation that can be used
        by a training agent."""

        if rnn_size is None and rnn_state is None:
            raise ValueError("one of rnn_state or rnn_size must be specified")
        if rnn_size is not None and rnn_state is None:
            rnn_state = rnn_placeholder_state(rnn_size)
        if rnn_size is not None and rnn_history is None:
            rnn_history = rnn_placeholder_state(rnn_size)
        action_mask = self.get_valid_moves(state)
        hint_mask = self.get_hint_mask(state)
        observation = state.to_observation(
            move_mask=action_mask,
            hint_mask=hint_mask,
            rnn_state=rnn_state,
            rnn_history=rnn_history,
        )
        return observation

    def get_win_signal(self, env_state: MathyEnvState) -> float:
        """Calculate the reward value for completing the episode. This is done
        so that the reward signal can be scaled based on the time it took to
        complete the episode. """
        tiny = 3e-10
        total_moves = max(tiny, env_state.max_moves)
        # guard against divide by zero with max and a small value
        current_move = max(tiny, total_moves - env_state.agent.moves_remaining)
        bonus = (total_moves / current_move) / total_moves
        # If the episode is not very short, and the agent completes in half
        # the number of allowed steps, double the bonus signal
        if total_moves > 4 and current_move < total_moves / 2:
            bonus *= 2
        return GameRewards.WIN + bonus

    def get_lose_signal(self, env_state: MathyEnvState) -> float:
        """Calculate the reward value for failing to complete the episode. This is done
        so that the reward signal can be problem-type dependent."""
        return GameRewards.LOSE

    def get_state_transition(
        self, env_state: MathyEnvState, searching: bool = False
    ) -> time_step.TimeStep:
        """Given an input state calculate the transition value of the timestep.

        This returns a nametuple provided by the `tf_agents` library.

        Input:
            env_state: current env_state
            searching: True when called by MCTS simulation

        Returns:
            transition: the current state value transition
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        features = env_state.to_observation(self.get_valid_moves(env_state))
        root = expression.get_root()

        # Subclass specific win conditions happen here. Custom win-conditions
        # outside of that can override this method entirely.
        result = self.transition_fn(env_state, root, features)
        if result is not None:
            return result

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, self.get_lose_signal(env_state))

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(
            sorted([f"{h.raw}" for h in env_state.agent.history])
        ):
            list_count = len(list(group))
            if list_count <= 1 or key != expression.raw:
                continue

            # NOTE: the reward is scaled by how many times this state has been visited
            #       up to (n) times
            multiplier = min(list_count, 3)
            return time_step.transition(
                features,
                reward=GameRewards.PREVIOUS_LOCATION * multiplier,
                discount=self.discount,
            )

        if len(agent.history) > 0:
            last_timestep = agent.history[-1]
            rule = self.get_rule_from_timestep(last_timestep)
            reward_actions = self.get_rewarding_actions(env_state)
            # The rewarding_actions can be user specified
            for rewarding_class in reward_actions:
                if isinstance(rule, rewarding_class):
                    return time_step.transition(
                        features,
                        reward=GameRewards.HELPFUL_MOVE,
                        discount=self.discount,
                    )

            penalty_actions = self.get_penalizing_actions(env_state)
            # The rewarding_actions can be user specified
            for penalty_class in penalty_actions:
                if isinstance(rule, penalty_class):
                    return time_step.transition(
                        features,
                        reward=GameRewards.UNHELPFUL_MOVE,
                        discount=self.discount,
                    )

        # We're in a new state, and the agent is a little older.
        return time_step.transition(
            features, reward=GameRewards.TIMESTEP, discount=self.discount
        )

    def get_next_state(
        self, env_state: MathyEnvState, action: int, searching: bool = False
    ) -> Tuple[MathyEnvState, time_step.TimeStep, ExpressionChangeRule]:
        """
        Input:
            env_state: current env_state
            action:    action taken
            searching: boolean set to True when called by MCTS

        Returns: tuple of
            next_state: env_state after applying action
            transition: the timestep that represents the state transition
            change: the change descriptor describing the change that happened
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        action_index, token_index = self.get_action_indices(action)
        token = self.get_token_at_index(expression, token_index)
        operation = self.rules[action_index]

        if (
            token is None
            or not isinstance(operation, BaseRule)
            or operation.can_apply_to(token) is False
        ):
            msg = "Invalid action({}) '{}' for expression '{}'."
            raise Exception(msg.format(action, type(operation), expression))

        change = operation.apply_to(token.clone_from_root())
        root = change.result.get_root()
        change_name = operation.name
        out_problem = str(root)
        out_env = env_state.encode_player(
            problem=out_problem,
            focus_index=token_index,
            action=action_index,
            moves_remaining=agent.moves_remaining - 1,
        )

        if not searching and self.verbose:
            token_idx = int("{}".format(token_index).zfill(3))
            self.print_state(out_env, change_name[:25].lower(), token_idx, change)
        transition = self.get_state_transition(out_env, searching)
        return out_env, transition, change

    def print_state(
        self,
        env_state: MathyEnvState,
        action_name: str,
        token_index: int = -1,
        change: ExpressionChangeRule = None,
        change_reward: float = 0.0,
    ):
        """Render the given state to stdout for visualization"""
        print(
            self.render_state(
                env_state, action_name, token_index, change, change_reward
            )
        )

    def render_state(
        self,
        env_state: MathyEnvState,
        action_name: str,
        token_index: int = -1,
        change: ExpressionChangeRule = None,
        change_reward: float = 0.0,
    ):
        """Render the given state to a string suitable for printing to a log"""
        changed_problem = env_state.agent.problem
        if change is not None:
            changed_problem = change.result.get_root().colored
        output = """{:<25} | {}""".format(action_name.lower(), changed_problem)

        def get_move_shortname(index, move):
            if move == 0:
                return "--"
            if move >= len(self.rules):
                return "xx"
            return self.rules[index].code.lower()

        token = "{}".format(token_index).zfill(3)
        moves_left = str(env_state.agent.moves_remaining).zfill(2)
        valid_rules = self.get_valid_rules(env_state)
        valid_moves = self.get_valid_moves(env_state)
        num_moves = "{}".format(len(np.nonzero(valid_moves)[0])).zfill(3)
        move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_rules)]
        moves = " ".join(move_codes)
        reward = f"{change_reward:.2}"
        reward = f"{reward:<5}"
        return f"{num_moves} | {moves} | {moves_left} | {token} | {reward} | {output}"

    def get_initial_state(
        self, params: Optional[MathyEnvProblemArgs] = None, print_problem: bool = True
    ) -> Tuple[MathyEnvState, MathyEnvProblem]:
        """Generate an initial MathyEnvState for an episode"""
        config = params if params is not None else MathyEnvProblemArgs()
        prob: MathyEnvProblem = self.problem_fn(config)
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()
        self.max_moves = self.max_moves_fn(prob, config)

        # Build and return the initial state
        env_state = MathyEnvState(
            problem=prob.text,
            problem_type=self.get_env_namespace(),
            max_moves=self.max_moves,
            num_rules=len(self.rules),
        )
        if print_problem and self.verbose:
            self.print_state(env_state, "initial-state")
        return env_state, prob

    def get_agent_actions_count(self, env_state: MathyEnvState) -> int:
        """Return number of all possible actions"""
        node_count = len(self.parser.parse(env_state.agent.problem).toList())
        return self.action_size * node_count

    def get_token_at_index(
        self, expression: MathExpression, focus_index: int
    ) -> Optional[MathExpression]:
        """Get the token that is `focus_index` from the left of the expression"""
        count = 0
        result = None

        def visit_fn(node, depth, data):
            nonlocal result, count
            result = node
            if count == focus_index:
                return STOP
            count = count + 1

        expression.visit_inorder(visit_fn)
        return result

    def get_valid_moves(self, env_state: MathyEnvState) -> List[int]:
        """Get a vector the length of the action space that is filled
         with 1/0 indicating whether the action at that index is valid
         for the current state.
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        return self.get_actions_for_node(expression)

    def get_hint_mask(self, env_state: MathyEnvState) -> List[int]:
        """Return a 0/1 mask of shape [len(actions) * len(nodes_in_expr)] that
        indicates nodes in the expression that the model should be hinted at
        may be salient to act on given the current task.

        The default implementation marks all the rules in any "term" node as
        possibly salient. A separate embedding is learned from this mask, and
        that embedding is used to attend to the input sequence.
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        node_list: List[MathExpression] = expression.toList()
        node_count = len(node_list)
        rule_count = len(self.rules)
        hints = [0] * rule_count * node_count
        for index, node in enumerate(node_list):
            term: Optional[TermEx] = get_term_ex(node)
            if term is None:
                continue
            # The move mask indicates valid node/rule combinations
            # for the hints we mark all rules on any term node
            for i in range(rule_count):
                hints[(index * rule_count) + i] = 1
        return hints

    def get_valid_rules(self, env_state: MathyEnvState) -> List[int]:
        """Get a vector the length of the number of valid rules that is
        filled with 0/1 based on whether the rule has any nodes in the
        expression that it can be applied to.

        NOTE: If you want to get a list of which nodes each rule can be
        applied to, prefer to use the `get_valid_moves` method.
        """
        key = self.to_hash_key(env_state)
        if key in self.valid_rules_cache:
            return self.valid_rules_cache[key]
        expression = self.parser.parse(env_state.agent.problem)
        actions = [0] * len(self.rules)
        for rule_index, rule in enumerate(self.rules):
            nodes = rule.find_nodes(expression)
            actions[rule_index] = 0 if len(nodes) == 0 else 1
        self.valid_rules_cache[key] = actions[:]
        return actions

    def get_action_indices(self, action: int) -> Tuple[int, int]:
        """Get the normalized action/node_index values from a
        given absolute action value.

        Returns a tuple of (rule_index, node_index)"""
        rule_count = len(self.rules)
        # Rule index = val % rule_count
        action_index = action % rule_count
        # And the action at that token
        token_index = int((action - action_index) / rule_count)
        return action_index, token_index

    def get_rule_from_timestep(self, time_step: MathyEnvTimeStep):
        return self.rules[time_step.action]

    def get_actions_for_node(self, expression: MathExpression) -> List[int]:
        key = str(expression)
        if key in self.valid_actions_mask_cache:
            return self.valid_actions_mask_cache[key][:]
        node_count = len(expression.toList())
        rule_count = len(self.rules)
        actions = [0] * rule_count * node_count
        for rule_index, rule in enumerate(self.rules):
            nodes = rule.find_nodes(expression)
            for node in nodes:
                action_index = (node.r_index * rule_count) + rule_index
                actions[action_index] = 1
        self.valid_actions_mask_cache[key] = actions[:]
        return actions

    def to_hash_key(self, env_state: MathyEnvState) -> str:
        """Convert env_state to a string for MCTS cache"""
        return env_state.agent.problem
