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

    def __init__(self, verbose=False, max_moves=None, lesson=None, focus_buckets=3):
        # To support arbitrary length inputs we bucket where to apply actions.
        self.focus_buckets = focus_buckets
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
        self.discount = 0.7
        self.verbose = verbose
        self.max_moves = max_moves if max_moves is not None else MathGame.max_moves_hard
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.lesson = lesson
        self.available_rules = [
            ConstantsSimplifyRule(),
            CommutativeSwapRule(),
            DistributiveMultiplyRule(),
            DistributiveFactorOutRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
        ]

        # We have number of actions * focus_buckets to support picking actions near
        # certain parts of an expression, without limiting sequence length or exploding
        # the action space by selecting from actions*max_supported_length actions.
        self._action_count = len(self.available_rules) * self.focus_buckets
        self.expression_str = "unset"

    @property
    def action_size(self):
        """Return the number of available actions"""
        return self._action_count

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

    def get_initial_state(self):

        if self.lesson is None:
            (problem, type, complexity) = self.problems.random_problem(
                [MODE_SIMPLIFY_POLYNOMIAL]
            )
        else:
            (problem, complexity) = self.lesson.problem_fn()
            type = self.lesson.problem_type
        self.expression_str = problem
        if self.verbose:
            print("\n\n[Problem] {}\n".format(problem))
        env_state = MathEnvironmentState(
            problem=problem, problem_type=type, max_moves=self.max_moves
        )
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
        return self.action_size

    def get_index_at_focus(self, expression: MathExpression, focus_value):
        """Given a 0-1 focus value, return the index of the node it nearest 
        points to in the expression when visited left-to-right."""
        count = 0

        def visit_fn(node, depth, data):
            nonlocal count
            count = count + 1

        expression.visitPreorder(visit_fn)
        return int(count * focus_value)

    def get_focus_from_action(self, action: int):
        """To support arbitrary length inputs while also letting the model select
        which node to apply which action to, we create (n) buckets of actions, and 
        then repeat the actions vector that many times. To derive a node to focus on
        we then take the selected action and determine which focus bucket it is in.
        
        Return a tuple of(focus_value, rule_index) for a given observed action """
        # Simple case in the first bucket
        if action < len(self.available_rules):
            return 0.0, action
        bucket_action = action % len(self.available_rules)
        bucket_number = (action - bucket_action) / len(self.available_rules)
        bucket_focus = bucket_number / (self.focus_buckets - 1)
        return bucket_focus, bucket_action

    def get_focus_at_index(
        self,
        env_state: MathEnvironmentState,
        action: int,
        expression: MathExpression = None,
    ):
        """Find the nearest actionable node index for the environment state 
        and given action.
        
        Returns: a tuple of (node_index, rule_instance) or (-1, None) if no 
                 applicable nodes are found
        """

        if expression is None:
            expression = self.parser.parse(env_state.agent.problem)
        # mod out the focus bucket
        action = action % len(self.available_rules)
        if action >= len(self.available_rules):
            pass
            f = 2
        rule = self.available_rules[action]
        if not isinstance(rule, BaseRule):
            raise ValueError("given action does not correspond to a BaseRule")

        # Select an actionable node around the agent focus
        focus = self.get_index_at_focus(expression, env_state.agent.focus)

        # Find the nearest node that can apply the given action
        possible_node_indices = [n.r_index for n in rule.findNodes(expression)]
        if len(possible_node_indices) == 0:
            return -1, None
        nearest_possible_index = min(
            possible_node_indices, key=lambda x: abs(x - focus)
        )
        return nearest_possible_index, rule

    def get_action_rule(
        self, env: MathEnvironmentState, expression: MathExpression, action: int
    ):

        if expression is None:
            expression = self.parser.parse(env_state.agent.problem)
        bucket_focus, bucket_action = self.get_focus_from_action(action)
        rule = self.available_rules[bucket_action]
        if not isinstance(rule, BaseRule):
            raise ValueError("given action does not correspond to a BaseRule")

        # Select an actionable node around the agent focus
        focus = self.get_index_at_focus(expression, bucket_focus)

        # Find the nearest node that can apply the given action
        possible_node_indices = [n.r_index for n in rule.findNodes(expression)]
        if len(possible_node_indices) == 0:
            return -1, None
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

        Returns: tuple of (next_state, reward, is_done)
            next_state: env_state after applying action
            reward: reward value for the action taken
            is_done: boolean indicating if the episode is done
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        operation, token = self.get_action_rule(env_state, expression, action)

        if not isinstance(operation, BaseRule) or not operation.canApplyTo(token):
            # operation, token = self.get_action_rule(env_state, expression, action)
            msg = "Invalid move selected ({}) for expression({}). Rule({}) does not apply."
            raise Exception(msg.format(action, expression, type(operation)))

        change = operation.applyTo(token.rootClone())
        root = change.result.getRoot()
        out_problem = str(root)
        out_env = env_state.encode_player(out_problem, agent.moves_remaining - 1)
        if not searching and self.verbose:
            output = """{:<25} | {}""".format(
                change.rule.name[:25].lower(), change.result.getRoot()
            )

            def get_move_shortname(index, move):
                if move == 0:
                    return "--"
                if move >= len(self.available_rules):
                    return "xx"
                return self.available_rules[index].code.lower()

            bucket, _ = self.get_focus_from_action(action)
            bucket = "{0:.1f}".format(bucket).zfill(3)
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

        actions = self.get_actions_for_expression(expression)
        # NOTE: Below is verbose output showing which actions are valid.
        # out_string = "{} :".format(agent.problem)
        # for i, action in enumerate(actions):
        #     if action == 0:
        #         continue
        #     out_string = out_string + " {}".format(self.available_rules[i].name)
        # print(out_string)
        return actions

    def get_actions_for_expression(self, expression: MathExpression):
        actions = [0] * self.action_size

        # Properties of numbers and common simplifications
        rule_count = len(self.available_rules)
        for rule_index, rule in enumerate(self.available_rules):
            nodes = rule.findNodes(expression)
            if len(nodes) > 0:
                for focus_bucket in range(self.focus_buckets):
                    actions[rule_count * focus_bucket + rule_index] = 1
            for node in nodes:
                token_index = node.r_index
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
            transition: the transition value for the current state
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
        for key, group in groupby(sorted(env_state.agent.history)):
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

        # # The agent is penalized for returning to a previous state.
        # for key, group in groupby(sorted(agent.history)):
        #     list_group = list(group)
        #     list_count = len(list_group)
        #     if list_count <= 1:
        #         continue
        #     # if not searching and self.verbose:
        #     #     print("\n[Penalty] re-entered previous state: {}".format(list_group[0]))
        #     return REWARD_PREVIOUS_LOCATION

        # # Check for problem_type specific win conditions
        # root = expression.getRoot()
        # if agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL and not has_like_terms(root):
        #     term_nodes = getTerms(root)
        #     is_win = True
        #     for term in term_nodes:
        #         if not isPreferredTermForm(term):
        #             is_win = False
        #     if is_win:
        #         if not searching and self.verbose:
        #             print(
        #                 "\n[Solved] {} => {}\n".format(self.expression_str, expression)
        #             )
        #         return REWARD_WIN

        # # Check the turn count last because if the previous move that incremented
        # # the turn over the count resulted in a win-condition, we want it to be honored.
        # if agent.moves_remaining <= 0:
        #     if not searching:
        #         self.write_draw(
        #             "[Failed] exhausted moves:\n\t input: {}\n\t 1: {}\n".format(
        #                 self.expression_str, expression
        #             )
        #         )
        #     return REWARD_LOSE

        # # The game continues at a reward cost of one per step
        # return REWARD_TIMESTEP

    def to_hash_key(self, env_state: MathEnvironmentState):
        """conversion of env_state to a string format, required by MCTS for hashing."""
        return str(env_state.agent.problem)


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
