import random
import math
import numpy
import time
import random
from alpha_zero_general.Game import Game
from .core.expressions import (
    MathExpression,
    ConstantExpression,
    MultiplyExpression,
    PowerExpression,
    STOP,
    AddExpression,
    VariableExpression,
)
from .core.problems import (
    ProblemGenerator,
    MODE_SOLVE_FOR_VARIABLE,
    MODE_SIMPLIFY_POLYNOMIAL,
)
from .core.parser import ExpressionParser
from .core.util import (
    termsAreLike,
    isAddSubtract,
    getTerms,
    getTerm,
    isPreferredTermForm,
    has_like_terms,
)
from .core.rules import (
    BaseRule,
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
    VariableMultiplyRule,
)
from .environment_state import (
    MathEnvironmentState,
    MathAgentState,
    MODEL_WIDTH,
    MODEL_HISTORY_LENGTH,
)
from multiprocessing import cpu_count
from itertools import groupby


class MathGame(Game):
    """
    Implement a math solving game where a player wins by executing the right sequence 
    of actions to reduce a math expression to an agreeable basic representation in as 
    few moves as possible.
    """

    width = MODEL_WIDTH
    history_length = MODEL_HISTORY_LENGTH
    # Default number of max moves used for training (can be overridden in init)
    max_moves_easy = 50
    max_moves_hard = 35
    max_moves_expert = 20

    def __init__(self, verbose=False, max_moves=None, training_wheels=True):
        self.verbose = verbose
        self.training_wheels = training_wheels
        self.max_moves = max_moves if max_moves is not None else MathGame.max_moves_hard
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.available_rules = [
            ConstantsSimplifyRule(),
            DistributiveFactorOutRule(),
            DistributiveMultiplyRule(),
            CommutativeSwapRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
        ]
        self.expression_str = "unset"

    def get_initial_state(self):
        (problem, type, complexity) = self.problems.random_problem(
            [MODE_SIMPLIFY_POLYNOMIAL]
        )
        # TODO: Remove this stateful variable that is used mostly for printing out "{from} -> {to}" at game end
        # NOTE: If we store a plane for history per user we could do something like [first_state, last_n-2, last_n-1, last_n, current]
        # problem = "(((10z + 8) + 11z * 4) + 1z) + 9z"
        # problem = "6z * 12x * 5 * 12x + 8z"
        # problem = "6z  + 2 - 5z"
        # problem = "-z + 6z"
        # problem = "5z + 6z"
        self.expression_str = problem
        # self.expression_str = "4x * 8 * 2"
        if self.verbose:
            print("\n\n[Problem] {}\n".format(problem))
        if len(list(problem)) > MathGame.width:
            raise ValueError(
                'Expression "{}" is too long for the current model to process. Max width is: {}'.format(
                    problem, MathGame.width
                )
            )
        env_state = MathEnvironmentState(problem=problem, problem_type=type)
        # NOTE: This is called for each episode, so it can be thought of like "onInitEpisode()"
        return env_state, complexity

    def write_draw(self, state):
        """Help spot errors in win conditons by always writing out draw values for review"""
        with open("draws.txt", "a") as file:
            file.write("{}\n".format(state))
        if self.verbose:
            print(state)

    def get_agent_actions_count(self):
        """Return number of all possible actions"""
        return len(self.available_rules)

    def get_action_rule(
        self, env_state: MathEnvironmentState, expression: MathExpression, action: int
    ):
        rule = self.available_rules[action]
        if not isinstance(rule, BaseRule):
            raise ValueError("given action does not correspond to a BaseRule")
        focus = random.randint(0, 3)
        # Find the nearest node that can apply the given action
        possible_node_indices = [n.r_index for n in rule.findNodes(expression)]
        nearest_possible_index = min(
            possible_node_indices, key=lambda x: abs(x - focus)
        )
        return rule, self.get_token_at_index(expression, nearest_possible_index)

    def get_next_state(self, env_state: MathEnvironmentState, action, searching=False):
        """
        Input:
            env_state: current env_state
            action:    action taken
            searching: boolean set to True when called by MCTS

        Returns:
            next_state: env_state after applying action
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        operation, token = self.get_action_rule(env_state, expression, action)

        # NOTE: If you get maximum recursion errors, it can mean that you're not
        # hitting a terminal state. Force the searching var off here to get
        # more verbose logging if your crash is occurring before the first verbose
        # output.
        # searching = False

        # Enforce constraints to keep training time and complexity down?
        # - can't commutative swap immediately to return to previous state.
        # NOTE: leaving these ideas here, but optimization made them less necessary
        # NOTE: Also adding constraints caused actions to be avoided and others to be
        #       repeated in odd ways. Assume repetition is part of training.
        # NOTE: Maybe this is solved by something like Actor/Critic updates?
        #
        # NOTE: This can maybe be solved by treating an expression returning to a previous
        #       state as a LOSS. If the model should optimize for the shortest paths
        #       to a solution, it will never be the shortest path if you revisit a previous
        #       state.
        # NOTE: The hope is that this will make the problem much simpler for the model
        if isinstance(operation, BaseRule) and operation.canApplyTo(token):
            change = operation.applyTo(token.rootClone())
            root = change.result.getRoot()
            out_problem = str(root)
            if not searching and self.verbose:
                output = """{:<25}: {}""".format(
                    change.rule.name[:25], change.result.getRoot()
                )
                print("[{}] {}".format(str(agent.move_count).zfill(2), output))
            out_env = env_state.encode_player(out_problem, agent.move_count + 1)
        else:
            print("action is {}, and token is {}".format(action, str(token)))
            raise Exception(
                "\n\n\tExpression: {}\n\tFocus: {}\n\tinvalid move selected: {}, {}".format(
                    expression, token, action, type(operation)
                )
            )

        return out_env

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

        expression.visitPreorder(visit_fn)
        return result

    def getValidMoves(self, env_state: MathEnvironmentState):
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

        actions = self.get_actions_for_expression(expression)
        # NOTE: Below is verbose output showing which actions are valid.
        # if not searching:
        #     print_list = self.available_actions + self.available_rules
        #     [
        #         print(
        #             "action[{}][{}] = {}".format(
        #                 i, bool(a), type(print_list[i]) if a != 0 else ""
        #             )
        #         )
        #         for i, a in enumerate(actions)
        #     ]
        return actions

    def get_actions_for_expression(self, expression: MathExpression):
        actions = [0] * self.get_agent_actions_count()

        # Properties of numbers and common simplifications
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.findNodes(expression)
            for node in nodes:
                token_index = node.r_index
                actions[rule_index] = 1
                # print(
                #     "[action_index={}={}] can apply to [token_index={}, {}]".format(
                #         action_index, rule.name, node.r_index, str(node)
                #     )
                # )
        return actions

    def getGameEnded(self, env_state: MathEnvironmentState, searching=False):
        """
        Input:
            env_state:     current env_state
            searching: boolean that is True when called by MCTS simulation

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)

        if self.training_wheels is True:
            # The player loses if they return to a previous state.
            for key, group in groupby(sorted(agent.history)):
                list_group = list(group)
                list_count = len(list_group)
                if list_count <= 1:
                    continue
                if not searching and self.verbose:
                    print(
                        "\n[Failed] re-entered previous state: {}".format(list_group[0])
                    )
                return -1

        # Check for problem_type specific win conditions
        root = expression.getRoot()
        if agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL and not has_like_terms(root):
            term_nodes = getTerms(root)
            is_win = True
            for term in term_nodes:
                if not isPreferredTermForm(term):
                    is_win = False
            if is_win:
                if not searching and self.verbose:
                    print(
                        "\n[Solved] {} => {}\n".format(self.expression_str, expression)
                    )
                return 1

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if agent.move_count > self.max_moves:
            if not searching:
                self.write_draw(
                    "[Failed] exhausted moves:\n\t input: {}\n\t 1: {}\n".format(
                        self.expression_str, expression
                    )
                )
            return -1

        # The game continues
        return 0

    def get_agent_state_size(self):
        """return shape (x,y) of agent state dimensions"""
        # (N+1, width) where N is history length and the additional row is for
        # game state.
        return (MathGame.history_length + 1, MathGame.width)

    def to_hash_key(self, env_state: MathEnvironmentState):
        """conversion of env_state to a string format, required by MCTS for hashing."""
        # return str(env_state.agent.problem)
        return "[{}, {}]".format(
            env_state.agent.problem, ", ".join(env_state.agent.history)
        )


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
