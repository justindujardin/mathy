from itertools import groupby
from typing import Optional, List, Type, Dict, Any, Tuple
from tf_agents.trajectories import time_step
from .types import MathyEnvProblem
from .core.expressions import STOP, MathExpression
from .core.parser import ExpressionParser
from .core.rules import (
    AssociativeSwapRule,
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
    ExpressionChangeRule,
)
from .mathy_env_state import MathyEnvTimeStep, MathyEnvState
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

    actions: List[BaseRule]
    rewarding_actions: List[Type[BaseRule]]
    max_moves: int
    verbose: bool
    reward_discount: float
    parser: ExpressionParser
    valid_actions_mask_cache: Dict[str, List[int]]
    valid_rules_cache: Dict[str, List[int]]

    INVALID_PROBLEM = MathyEnvProblem("invalid", -1, -1)

    def __init__(
        self,
        actions=None,
        rewarding_actions=None,
        max_moves=20,
        verbose=False,
        reward_discount=0.99,
    ):
        self.discount = reward_discount
        self.verbose = verbose
        self.max_moves = max_moves
        self.parser = ExpressionParser()
        self.actions = actions
        if self.actions is None:
            self.actions = mathy_core_rules()
        self.rewarding_actions = rewarding_actions
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()

    @property
    def action_size(self) -> int:
        """Return the number of available actions"""
        return len(self.actions)

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        """Get the list of rewarding action types. When these actions
        are selected, the agent gets a positive reward as opposed to the
        normal negative timestep reward."""
        return []

    def transition_fn(
        self, env_state: MathyEnvState, expression: MathExpression, features: Any
    ) -> Optional[time_step.TimeStep]:
        """Provide environment-specific transitions per timestep."""
        return None

    def problem_fn(self, params: Dict[str, Any] = None) -> MathyEnvProblem:
        """Return a problem for the environment given an optional set
        of parameters to control problem generation. This is implemented
        per environment such that each environment can generate its own
        dataset with no required configuration. """
        return MathyEnv.INVALID_PROBLEM

    def get_win_signal(self, env_state: MathyEnvState) -> float:
        """Calculate the reward value for completing the episode. This is done
        so that the reward signal can be scaled based on the time it took to
        complete the episode. """
        tiny = 3e-10
        total_moves = max(tiny, env_state.max_moves)
        # guard against divide by zero with max and a small value
        current_move = max(tiny, total_moves - env_state.agent.moves_remaining)
        bonus = (total_moves / current_move) / total_moves
        # If the agent completes in half the allowed steps, double the bonus signal
        if current_move < total_moves / 2:
            bonus *= 2
        return GameRewards.WIN + bonus

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
        features = env_state.to_input_features(self.get_valid_moves(env_state))
        root = expression.get_root()

        # Subclass specific win conditions happen here. Custom win-conditions
        # outside of that can override this method entirely.
        result = self.transition_fn(env_state, root, features)
        if result is not None:
            return result

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, GameRewards.LOSE)

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

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(
            sorted([f"{h.raw}" for h in env_state.agent.history])
        ):
            list_group = list(group)
            list_count = len(list_group)
            if list_count <= 1:
                continue

            return time_step.transition(
                features, reward=GameRewards.PREVIOUS_LOCATION, discount=self.discount
            )

        # We're in a new state, and the agent is a little older.
        return time_step.transition(
            features, reward=GameRewards.TIMESTEP, discount=self.discount
        )

    def get_next_state(
        self, env_state: MathyEnvState, action: int, searching: bool = False
    ) -> Tuple[MathyEnvState, time_step.TimeStep]:
        """
        Input:
            env_state: current env_state
            action:    action taken
            searching: boolean set to True when called by MCTS

        Returns: tuple of (next_state, reward, is_done)
            next_state: env_state after applying action
            reward: reward value for the action taken
            is_done: boolean indicating if the episode is done
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        action_index, token_index = self.get_action_indices(action)
        token = self.get_token_at_index(expression, token_index)
        operation = self.actions[action_index]

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
        return out_env, transition

    def print_state(
        self,
        env_state: MathyEnvState,
        action_name: str,
        token_index: int = -1,
        change: ExpressionChangeRule = None,
    ):
        """Render the given state to stdout for visualization"""
        changed_problem = env_state.agent.problem
        if change is not None:
            changed_problem = change.result.get_root().colored
        output = """{:<25} | {}""".format(action_name.lower(), changed_problem)

        def get_move_shortname(index, move):
            if move == 0:
                return "--"
            if move >= len(self.actions):
                return "xx"
            return self.actions[index].code.lower()

        token_idx = "{}".format(token_index).zfill(3)
        moves_left = str(env_state.agent.moves_remaining).zfill(2)
        valid_moves = self.get_valid_rules(env_state)
        move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_moves)]
        moves = " ".join(move_codes)
        print("{} | {} | {} | {}".format(moves, moves_left, token_idx, output))

    def get_initial_state(
        self, params: Dict[str, Any] = None, print_problem: bool = True
    ) -> Tuple[MathyEnvState, MathyEnvProblem]:
        """Generate an initial MathyEnvState with the game's configuration"""
        prob = self.problem_fn(params)
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()
        turns_preference = (
            None if params is None else params.get("turns_per_complexity", None)
        )
        # If a turns per complexity value is passed, adjust max moves for
        # the problem output
        if turns_preference is not None:
            self.max_moves = turns_preference * prob.complexity

        # Build and return the initial state
        env_state = MathyEnvState(
            problem=prob.text, problem_type=prob.type, max_moves=self.max_moves
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
        actions = [0] * len(self.actions)
        for rule_index, rule in enumerate(self.actions):
            nodes = rule.find_nodes(expression)
            actions[rule_index] = 0 if len(nodes) == 0 else 1
        self.valid_rules_cache[key] = actions[:]
        return actions

    def get_action_indices_from_timestep(
        self, time_step: MathyEnvTimeStep
    ) -> Tuple[int, int]:
        """Parse a timestep and return the unpacked action_index/token_index
        from the source action"""
        return time_step.action, time_step.focus

    def get_action_indices(self, action: int) -> Tuple[int, int]:
        """Get the normalized action/node_index values from a
        given absolute action value.

        Returns a tuple of (rule_index, node_index)"""
        rule_count = len(self.actions)
        # Rule index = val % rule_count
        action_index = action % rule_count
        # And the action at that token
        token_index = int((action - action_index) / rule_count)
        return action_index, token_index

    def get_rule_from_timestep(self, time_step: MathyEnvTimeStep):
        return self.actions[time_step.action]

    def get_actions_for_node(self, expression: MathExpression) -> List[int]:
        key = str(expression)
        if key in self.valid_actions_mask_cache:
            return self.valid_actions_mask_cache[key][:]
        node_count = len(expression.toList())
        rule_count = len(self.actions)
        actions = [0] * rule_count * node_count
        for rule_index, rule in enumerate(self.actions):
            nodes = rule.find_nodes(expression)
            for node in nodes:
                action_index = (node.r_index * rule_count) + rule_index
                actions[action_index] = 1
        self.valid_actions_mask_cache[key] = actions[:]
        return actions

    def to_hash_key(self, env_state: MathyEnvState) -> str:
        """Convert env_state to a string for MCTS cache"""
        return env_state.agent.problem
