from itertools import groupby

from tf_agents.trajectories import time_step

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
)
from .core.util import get_terms, has_like_terms, is_preferred_term_form
from .environment_state import AgentTimeStep, MathEnvironmentState
from .game_modes import MODE_SIMPLIFY_POLYNOMIAL
from .util import GameRewards


class MathGame:
    """
    Implement a math solving game where a player wins by executing the right sequence
    of actions to reduce a math expression to an agreeable basic representation in as
    few moves as possible.
    """

    def __init__(
        self,
        verbose=False,
        max_moves=20,
        lesson=None,
        reward_discount=0.99,
        rewarding_actions=None,
    ):
        self.discount = reward_discount
        self.verbose = verbose
        self.max_moves = max_moves
        self.parser = ExpressionParser()
        self.lesson = lesson
        self.available_rules = [
            ConstantsSimplifyRule(),
            CommutativeSwapRule(preferred=False),
            DistributiveMultiplyRule(),
            DistributiveFactorOutRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
        ]
        self.rewarding_actions = rewarding_actions
        if self.rewarding_actions is None:
            self.rewarding_actions = [ConstantsSimplifyRule, DistributiveFactorOutRule]

    @property
    def action_size(self):
        """Return the number of available actions"""
        return len(self.available_rules)

    def get_state_value(self, env_state: MathEnvironmentState, searching=False):
        """Get the value of the current state


        Input:
            env_state:     current env_state
            searching: boolean that is True when called by MCTS simulation

        Returns:
            transition: the current state value transition
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        features = env_state.to_input_features(self.get_valid_moves(env_state))
        root = expression.get_root()
        if (
            env_state.agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL
            and not has_like_terms(root)
        ):
            term_nodes = get_terms(root)
            is_win = True
            for term in term_nodes:
                if not is_preferred_term_form(term):
                    is_win = False
            if is_win:
                return time_step.termination(features, GameRewards.WIN)

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, GameRewards.LOSE)

        if len(agent.history) > 0:
            last_timestep = agent.history[-1]
            rule = self.get_rule_from_timestep(last_timestep)
            # The rewarding_actions can be user specified
            for rewarding_class in self.rewarding_actions:
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

    def get_next_state(self, env_state: MathEnvironmentState, action, searching=False):
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
        rule_count = len(self.available_rules)
        action_index, token_index = self.get_action_indices(action)
        token = self.get_token_at_index(expression, token_index)
        if action_index > rule_count - 1:
            operation = "dangit!"
        operation = self.available_rules[action_index]

        if (
            not isinstance(operation, BaseRule)
            or operation.can_apply_to(token) is False
        ):
            msg = "Invalid move '{}' for expression '{}'."
            raise Exception(msg.format(action, expression, type(operation)))

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
            output = """{:<25} | {}""".format(change_name[:25].lower(), out_problem)

            def get_move_shortname(index, move):
                if move == 0:
                    return "--"
                if move >= len(self.available_rules):
                    return "xx"
                return self.available_rules[index].code.lower()

            token_idx = "{}".format(token_index).zfill(3)
            moves_left = str(agent.moves_remaining).zfill(2)
            valid_moves = self.get_valid_rules(out_env)
            move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_moves)]
            moves = " ".join(move_codes)
            print("{} | {} | {} | {}".format(moves, moves_left, token_idx, output))
        transition = self.get_state_value(out_env, searching)
        return out_env, transition

    def print_state(
        self, env_state: MathEnvironmentState, action_name: str, token_index=-1
    ):
        """Render the given state to stdout for visualization"""
        output = """{:<25} | {}""".format(action_name.lower(), env_state.agent.problem)

        def get_move_shortname(index, move):
            if move == 0:
                return "--"
            if move >= len(self.available_rules):
                return "xx"
            return self.available_rules[index].code.lower()

        token_idx = "{}".format(token_index).zfill(3)
        moves_left = str(env_state.agent.moves_remaining).zfill(2)
        valid_moves = self.get_valid_rules(env_state)
        move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_moves)]
        moves = " ".join(move_codes)
        print("{} | {} | {} | {}".format(moves, moves_left, token_idx, output))

    def get_initial_state(self, print_problem=True):
        """Generate an initial MathEnvironmentState with the game's configuration"""
        if self.lesson is None:
            raise ValueError("cannot generate problems without a lesson plan")
        (problem, complexity) = self.lesson.problem_fn()
        type = self.lesson.problem_type
        env_state = MathEnvironmentState(
            problem=problem, problem_type=type, max_moves=self.max_moves
        )
        if print_problem and self.verbose:
            self.print_state(env_state, "initial-state")
        return env_state, complexity

    def get_agent_actions_count(self, env_state: MathEnvironmentState):
        """Return number of all possible actions"""
        node_count = len(self.parser.parse(env_state.agent.problem).toList())
        return self.action_size * node_count

    def get_token_at_index(
        self, expression: MathExpression, focus_index: int
    ) -> MathExpression:
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

    def get_valid_moves(self, env_state: MathEnvironmentState):
        """Get a vector the length of the action space that is filled
         with 1/0 indicating whether the action at that index is valid
         for the current state.
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        actions = self.get_actions_for_node(expression)
        return actions

    def get_valid_rules(self, env_state: MathEnvironmentState):
        """Get a vector the length of the number of valid rules that is
        filled with 0/1 based on whether the rule has any nodes in the
        expression that it can be applied to.

        NOTE: If you want to get a list of which nodes each rule can be 
        applied to, prefer to use the `get_valid_moves` method.
        """
        expression = self.parser.parse(env_state.agent.problem)
        actions = [0] * len(self.available_rules)
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.find_nodes(expression)
            actions[rule_index] = 0 if len(nodes) == 0 else 1
        return actions

    def get_action_indices_from_timestep(self, time_step: AgentTimeStep):
        """Parse a timestep and return the unpacked action_index/token_index
        from the source action"""
        return time_step.action, time_step.focus

    def get_action_indices(self, action: int):
        """Get the normalized action/node_index values from a
        given absolute action value.

        Returns a tuple of (rule_index, node_index)"""
        rule_count = len(self.available_rules)
        # Rule index = val % rule_count
        action_index = action % rule_count
        # And the action at that token
        token_index = int((action - action_index) / rule_count)
        return action_index, token_index

    def get_rule_from_timestep(self, time_step: AgentTimeStep):
        return self.available_rules[time_step.action]

    def get_actions_for_node(self, expression: MathExpression):
        node_count = len(expression.toList())
        rule_count = len(self.available_rules)
        actions = [0] * rule_count * node_count

        # Properties of numbers and common simplifications
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.find_nodes(expression)
            for node in nodes:
                action_index = (node.r_index * rule_count) + rule_index
                actions[action_index] = 1

                # print(
                #     "[action_index={}={}] can apply to [token_index={}, {}]".format(
                #         action_index, rule.name, node.r_index, str(node)
                #     )
                # )

        return actions

    def to_hash_key(self, env_state: MathEnvironmentState):
        """conversion of env_state to a string format, required by MCTS for hashing."""
        return f"[{env_state.agent.focus_index}]{env_state.agent.problem}"
