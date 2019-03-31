import math
from itertools import groupby
from multiprocessing import cpu_count
from ..core.expressions import STOP, MathExpression
from ..core.parser import ExpressionParser
from ..core.rules import (
    AssociativeSwapRule,
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)
from ..core.util import getTerms, has_like_terms, isPreferredTermForm
from ..environment_state import MathEnvironmentState
from ..training.problems import MODE_SIMPLIFY_POLYNOMIAL, ProblemGenerator
from ..util import (
    REWARD_LOSE,
    REWARD_PREVIOUS_LOCATION,
    REWARD_INVALID_ACTION,
    REWARD_NOT_HELPFUL_MOVE,
    REWARD_TIMESTEP,
    REWARD_NEW_LOCATION,
    REWARD_WIN,
    is_terminal_transition,
)
from tf_agents.environments import time_step
from ..agent_actions import VisitBeforeAction, VisitAfterAction, MetaAction


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
        # Tuned this by comparing a bunch of discounts. This allows for decaying of
        # the reward rapidly so we don't end up giving a bunch of value to a string
        # of stupid moves that finally gets to a good move (e.g. commutativ swap 10x
        # then finally constant arithmetic would lead to CA getting 1.0 and 10x commutative
        # swaps getting near 1.0) This kind of skew will only encourage the model to prefer
        # those more commonly used actions (even if they're not often the most valuable)
        #
        # This value works well for at least some postitive and negative outcomes:
        #
        # >>> discount([-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,
        #               -0.01,-0.01,-0.06,-0.06,-1.0],0.7)
        # array([-0.041066  , -0.04438   , -0.04911428, -0.05587755, -0.06553935,
        #        -0.07934193, -0.09905991, -0.12722844, -0.1674692 , -0.224956  ,
        #        -0.30708   , -0.4244    , -0.592     , -0.76      , -1.        ],
        #       dtype=float32)
        # >>> discount([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 1.0],0.7)
        # array([0.0882373, 0.140339 , 0.21477  , 0.3211   , 0.473    , 0.69     ,
        #        1.       ], dtype=float32)
        # UPDATE: setting back to 0.99 because the eps are longer when manipulating
        #         focus token explicitly
        self.discount = 0.99
        self.verbose = verbose
        self.max_moves = max_moves if max_moves is not None else MathGame.max_moves_hard
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.lesson = lesson
        self.available_rules = [
            VisitBeforeAction(),
            VisitAfterAction(),
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

    def get_initial_state(self, print_problem=True):

        if self.lesson is None:
            (problem, type, complexity) = self.problems.random_problem(
                [MODE_SIMPLIFY_POLYNOMIAL]
            )
        else:
            (problem, complexity) = self.lesson.problem_fn()
            type = self.lesson.problem_type
        self.expression_str = problem
        if print_problem and self.verbose:
            print("\n\n[Problem] {}\n".format(problem))
        env_state = MathEnvironmentState(
            problem=problem, problem_type=type, max_moves=self.max_moves
        )
        return env_state, complexity

    def write_draw(self, state):
        """Help spot errors in win conditons by always writing out draw values for review"""
        with open("draws.txt", "a") as file:
            file.write("{}\n".format(state))
        if self.verbose:
            print(state)

    def get_agent_actions_count(self):
        """Return number of all possible actions"""
        return self.action_size

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
        operation = self.available_rules[action]

        if isinstance(operation, MetaAction):
            # Move the agent focus around in the expression
            out_focus = operation.visit(self, expression, agent.focus_index)
            out_problem = str(expression)
            change_name = operation.__class__.__name__
        elif isinstance(operation, BaseRule):
            token = self.get_token_at_index(expression, agent.focus_index)
            if operation.canApplyTo(token) is False:
                msg = "Invalid move selected ({}) for expression({}). Rule({}) does not apply."
                raise Exception(msg.format(action, expression, type(operation)))
            change = operation.applyTo(token.rootClone())
            root = change.result.getRoot()
            change_name = operation.name
            out_problem = str(root)

            # NOTE: using rule.findNodes to mark the index of this node for use as new focus
            change.rule.findNodes(root)
            out_focus_node = change.result
            if change.focus_node is not None:
                out_focus_node = change.focus_node
            out_focus = out_focus_node.r_index

        out_env = env_state.encode_player(
            problem=out_problem,
            focus_index=out_focus,
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

            bucket = "{}".format(agent.focus_index).zfill(3)
            moves_left = str(agent.moves_remaining).zfill(2)
            valid_moves = self.get_valid_moves(out_env)[: len(self.available_rules)]
            move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_moves)]
            moves = " ".join(move_codes)
            print("{} | {} | {} | {}".format(moves, moves_left, bucket, output))
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

        expression.visitPreorder(visit_fn)
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
        focus_node = self.get_token_at_index(expression, agent.focus_index)
        actions = self.get_actions_for_node(focus_node)
        return actions

    def get_actions_for_node(self, expression: MathExpression):
        actions = [0] * self.action_size
        for index, action in enumerate(self.available_rules):
            if isinstance(action, MetaAction):
                actions[index] = 1
            elif isinstance(action, BaseRule) and action.canApplyTo(expression):
                actions[index] = 1
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
                return time_step.termination(features, REWARD_WIN)

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, REWARD_LOSE)

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(
            sorted([f"{h.raw}{h.focus}" for h in env_state.agent.history])
        ):
            list_group = list(group)
            list_count = len(list_group)
            if list_count <= 1:
                continue

            return time_step.transition(
                features, reward=REWARD_PREVIOUS_LOCATION, discount=self.discount
            )

        # We're in a new state, and the agent is a little older. Minus reward!
        return time_step.transition(
            features, reward=REWARD_TIMESTEP, discount=self.discount
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
