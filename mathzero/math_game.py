import math
from itertools import groupby
from multiprocessing import cpu_count
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
from .core.util import getTerms, has_like_terms, isPreferredTermForm
from .environment_state import MathEnvironmentState, AgentTimeStep
from .training.problems import MODE_SIMPLIFY_POLYNOMIAL, ProblemGenerator
from .util import GameRewards, is_terminal_transition
from tf_agents.environments import time_step


class MathGame:
    """
    Implement a math solving game where a player wins by executing the right sequence 
    of actions to reduce a math expression to an agreeable basic representation in as 
    few moves as possible.
    """

    # Default number of max moves used for training (can be overridden in init)
    max_moves_easy = 50
    max_moves_hard = 35
    max_moves_expert = 20

    def __init__(self, verbose=False, max_moves=None, lesson=None):

        self.discount = 0.99
        self.verbose = verbose
        self.max_moves = max_moves if max_moves is not None else MathGame.max_moves_hard
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.lesson = lesson
        self.available_rules = [
            ConstantsSimplifyRule(),
            CommutativeSwapRule(preferred=False),
            DistributiveMultiplyRule(),
            DistributiveFactorOutRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
        ]
        self.expression_str = "unset"

    @property
    def action_size(self):
        """Return the number of available actions"""
        return len(self.available_rules)

    def get_gpu_fraction(self):
        """
        Returns:
            gpu_fraction: the fraction of GPU memory to dedicate to the 
                          neural network for this game instance.
        """
        # NOTE: we double the CPU count to start out allocating smaller amounts of memory.
        #       This is because if we oversubscribe CUDA can throw failed to allocate errors
        #       with a bunch of workers. This way Tensorflow will grow the allocation per worker
        #       only as needed.
        return 1 / (cpu_count() * 1.5)

    def print_state(
        self, env_state: MathEnvironmentState, action_name: str, token_index=-1
    ):
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

        if self.lesson is None:
            (problem, type, complexity) = self.problems.random_problem(
                [MODE_SIMPLIFY_POLYNOMIAL]
            )
        else:
            (problem, complexity) = self.lesson.problem_fn()
            type = self.lesson.problem_type
        self.expression_str = problem
        env_state = MathEnvironmentState(
            problem=problem, problem_type=type, max_moves=self.max_moves
        )
        if print_problem and self.verbose:
            self.print_state(env_state, "initial-state")
        return env_state, complexity

    def write_draw(self, state):
        """Help spot errors in win conditons by always writing out draw values for review"""
        with open("draws.txt", "a") as file:
            file.write("{}\n".format(state))
        if self.verbose:
            print(state)

    def get_agent_actions_count(self, env_state: MathEnvironmentState):
        """Return number of all possible actions"""
        node_count = len(self.parser.parse(env_state.agent.problem).toList())
        return self.action_size * node_count

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
        total_actions = len(expression.toList()) * rule_count
        action_index, token_index = self.get_action_indices(expression, action)
        token = self.get_token_at_index(expression, token_index)
        if action_index > rule_count - 1:
            operation = "dangit!"
        operation = self.available_rules[action_index]

        if not isinstance(operation, BaseRule) or operation.canApplyTo(token) is False:
            msg = "Invalid move selected ({}) for expression({}). Rule({}) does not apply."
            raise Exception(msg.format(action, expression, type(operation)))

        change = operation.applyTo(token.rootClone())
        root = change.result.getRoot()
        change_name = operation.name
        out_problem = str(root)
        out_env = env_state.encode_player(
            problem=out_problem,
            focus_index=0,
            action=action,
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

        expression.visitInorder(visit_fn)
        return result

    def get_valid_moves(self, env_state: MathEnvironmentState):
        """
        Input:
            env_state: current env_state

        Returns:
            validMoves: a binary vector of length self.get_agent_actions_count(), 1 for
                        moves that are valid from the current env_state, and 0 for invalid 
                        moves
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        actions = self.get_actions_for_node(expression)
        return actions

    def get_valid_rules(self, env_state: MathEnvironmentState):
        """
        Input:
            env_state: current env_state

        Returns:
            validRules: a binary vector of length len(self.available_rules) for rules that can
                        be validly applied to at least one node in the current state.
        """
        expression = self.parser.parse(env_state.agent.problem)
        actions = [0] * len(self.available_rules)
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.findNodes(expression)
            actions[rule_index] = 0 if len(nodes) == 0 else 1
        return actions

    def get_action_indices_from_timestep(self, time_step: AgentTimeStep):
        """Parse a timestep and return the unpacked action_index/token_index from the source action"""
        expression = self.parser.parse(time_step.raw)
        return self.get_action_indices(expression, time_step.action)

    def get_action_indices(self, expression: MathExpression, action: int):
        rule_count = len(self.available_rules)
        nodes = expression.toList()
        node_count = len(nodes)
        total_actions = node_count * rule_count
        # Rule index = val % rule_count
        action_index = action % rule_count
        # And the action at that token
        token_index = int((action - action_index) / rule_count)
        return action_index, token_index

    def get_rule_from_timestep(self, time_step: AgentTimeStep):
        action_index, token_index = self.get_action_indices_from_timestep(time_step)
        return self.available_rules[action_index]

    def get_actions_for_node(self, expression: MathExpression):
        node_count = len(expression.toList())
        rule_count = len(self.available_rules)
        actions = [0] * rule_count * node_count

        # Properties of numbers and common simplifications
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.findNodes(expression)
            for node in nodes:
                action_index = (node.r_index * rule_count) + rule_index
                actions[action_index] = 1

                # print(
                #     "[action_index={}={}] can apply to [token_index={}, {}]".format(
                #         action_index, rule.name, node.r_index, str(node)
                #     )
                # )

        return actions

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
        features = env_state.to_input_features()
        root = expression.getRoot()
        if (
            env_state.agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL
            and not has_like_terms(root)
        ):
            term_nodes = getTerms(root)
            is_win = True
            for term in term_nodes:
                if not isPreferredTermForm(term):
                    is_win = False
            if is_win:
                return time_step.termination(features, GameRewards.WIN)

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, GameRewards.LOSE)

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(
            sorted([f"{h.raw}{h.focus}" for h in env_state.agent.history])
        ):
            list_group = list(group)
            list_count = len(list_group)
            if list_count <= 1:
                continue

            return time_step.transition(
                features, reward=GameRewards.PREVIOUS_LOCATION, discount=self.discount
            )

        # NOTE: perhaps right here is a good point for an abstraction around custom rules
        #       that generate different agent reward values based on game types.
        if len(agent.history) > 0:
            last_timestep = agent.history[-1]
            rule = self.get_rule_from_timestep(last_timestep)
            if isinstance(rule, ConstantsSimplifyRule):
                return time_step.transition(
                    features, reward=GameRewards.HELPFUL_MOVE, discount=self.discount
                )

        # We're in a new state, and the agent is a little older. Minus reward!
        return time_step.transition(
            features, reward=GameRewards.TIMESTEP, discount=self.discount
        )

    def to_hash_key(self, env_state: MathEnvironmentState):
        """conversion of env_state to a string format, required by MCTS for hashing."""
        return f"[{env_state.agent.focus_index}]{env_state.agent.problem}"


_parser = None


def display(env_state):
    global _parser
    if _parser is None:
        _parser = ExpressionParser()
    agent = env_state.agent
    expression = _parser.parse(agent.problem)
    expression_len = len(str(expression))
    width = 100
    buffer = " " * int(width / 2 - expression_len)
    print("{}{}".format(buffer, expression))
