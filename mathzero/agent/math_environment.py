import numpy
import tensorflow as tf
from tf_agents.environments import py_environment, time_step, utils, wrappers
from tf_agents.specs import array_spec, tensor_spec
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
from .math_embeddings import MathEmbeddings
from ..util import (
    REWARD_LOSE,
    REWARD_PREVIOUS_LOCATION,
    REWARD_INVALID_ACTION,
    REWARD_TIMESTEP,
    REWARD_WIN,
    is_terminal_reward,
)
from ..model.features import (
    FEATURE_TOKEN_VALUES,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_TOKEN_TYPES,
)

tf.compat.v1.enable_v2_behavior()


class MathEnvironment(py_environment.PyEnvironment):
    """
    Implement a math solving game where an agent wins by executing the right sequence 
    of actions to reduce a math expression to an agreeable target representation.
    """

    def __init__(self, lesson=None, max_moves=None, verbose=False):
        self.model_dir = "./training/embeddings"
        self.verbose = verbose
        self.discount = 1.0
        self.max_moves = max_moves if max_moves is not None else 50
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.lesson = lesson
        self.available_rules = [
            ConstantsSimplifyRule(),
            DistributiveFactorOutRule(),
            DistributiveMultiplyRule(),
            CommutativeSwapRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
        ]
        # Focus buckets
        self._focus_buckets = 3

        # We have number of actions * focus_buckets to support picking actions near
        # certain parts of an expression, without limiting sequence length or exploding
        # the action space by selecting from actions*max_supported_length actions.
        self._action_count = len(self.available_rules) * self._focus_buckets
        self.embedding_dimensions = 32
        self.embedding_model = MathEmbeddings(
            self._action_count, self.model_dir, self.embedding_dimensions
        )
        self.embedding_model.start()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=numpy.int32,
            minimum=0,
            maximum=self._action_count,
            name="action",
        )
        self._observation_spec = tensor_spec.TensorSpec(
            shape=(self.embedding_dimensions,), dtype=numpy.float32, name="embeddings"
        )
        self._state = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.get_initial_state()
        self._episode_ended = False
        return time_step.restart(self.embedding_model.predict(self._state))

    def _step(self, action):
        searching = False
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Assert action valid
        agent = self._state.agent
        expression = self.parser.parse(agent.problem)
        operation, token = self.get_action_rule(expression, action)

        if not isinstance(operation, BaseRule) or not operation.canApplyTo(token):
            # TODO: How to use MaskedDistribution with SAC agent? For now penalize to keep things moving.
            msg = "Invalid move selected ({}) for expression({}) with focus({}). Rule({}) does not apply."

            self._state = self._state.encode_player(
                agent.problem, agent.moves_remaining - 1
            )
            out_features = self.embedding_model.predict(self._state)
            if self._state.agent.moves_remaining <= 0:
                self._episode_ended = True
                return time_step.termination(out_features, REWARD_LOSE)
            res = time_step.transition(
                out_features, reward=REWARD_INVALID_ACTION, discount=self.discount
            )
            return res

        change = operation.applyTo(token.rootClone())
        root = change.result.getRoot()
        out_problem = str(root)
        if self.verbose:
            output = """{:<25}: {}""".format(
                change.rule.name[:25], change.result.getRoot()
            )
            print("[{}] {}".format(str(agent.moves_remaining).zfill(2), output))
        self._state = self._state.encode_player(out_problem, agent.moves_remaining - 1)
        out_features = self.embedding_model.predict(self._state)
        expression = self.parser.parse(self._state.agent.problem)

        # Check for problem_type specific win conditions
        root = expression.getRoot()
        if (
            self._state.agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL
            and not has_like_terms(root)
        ):
            term_nodes = getTerms(root)
            is_win = True
            for term in term_nodes:
                if not isPreferredTermForm(term):
                    is_win = False
            if is_win:
                self._episode_ended = True
                print("\n[Solved] {} => {}\n".format(self.expression_str, expression))
                return time_step.termination(out_features, REWARD_WIN)

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if self._state.agent.moves_remaining <= 0:
            self._episode_ended = True
            return time_step.termination(out_features, REWARD_LOSE)

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(sorted(self._state.agent.history)):
            list_group = list(group)
            list_count = len(list_group)
            if list_count <= 1:
                continue
            # if not searching and self.verbose:
            #     print("\n[Penalty] re-entered previous state: {}".format(list_group[0]))

            return time_step.transition(
                out_features, reward=REWARD_PREVIOUS_LOCATION, discount=self.discount
            )

        # The game continues at a reward cost of one per step
        return time_step.transition(
            out_features, reward=REWARD_TIMESTEP, discount=self.discount
        )

    def get_initial_state(self):
        """Generate a new problem and return an initial observation of the environment"""
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
        return env_state

    def get_agent_actions_count(self):
        """Return number of all possible actions"""
        return len(self.available_rules)

    def get_focus_example_from_token_index(
        self, expression: MathExpression, focus_index
    ):
        """Given an index into an expression, get 0-1 focus value
        that points back to that token, for model input data.
        """
        count = 0
        result = None

        def visit_fn(node, depth, data):
            nonlocal count
            count = count + 1

        expression.visitPreorder(visit_fn)
        return focus / count

    def get_token_index_from_focus(self, expression: MathExpression, focus_value):
        """Given a 0-1 focus value, return the index of the node it nearest 
        points to in the expression when visited left-to-right."""
        count = 0

        def visit_fn(node, depth, data):
            nonlocal count
            count = count + 1

        expression.visitPreorder(visit_fn)
        return int(count * focus_value)

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
        rule = self.available_rules[action]
        if not isinstance(rule, BaseRule):
            raise ValueError("given action does not correspond to a BaseRule")

        # Select an actionable node around the agent focus
        focus = self.get_token_index_from_focus(expression, env_state.agent.focus)

        # Find the nearest node that can apply the given action
        possible_node_indices = [n.r_index for n in rule.findNodes(expression)]
        if len(possible_node_indices) == 0:
            return -1, None
        nearest_possible_index = min(
            possible_node_indices, key=lambda x: abs(x - focus)
        )
        return nearest_possible_index, rule

    def get_action_rule(self, expression: MathExpression, action: int):
        env: MathEnvironment = self._state
        if expression is None:
            expression = self.parser.parse(env_state.agent.problem)
        bucket_action = action % len(self.available_rules)
        bucket_focus = (action - bucket_action) / self._focus_buckets
        # If we're not in bucket 0, divide by number of buckets to get
        # a value in the range 0-1 which points left-to-right in the expression
        # for where to apply the action
        if bucket_focus > 0.0:
            bucket_focus = (bucket_focus - 1) * (1 / self._focus_buckets)
        rule = self.available_rules[bucket_action]
        if not isinstance(rule, BaseRule):
            raise ValueError("given action does not correspond to a BaseRule")

        # Select an actionable node around the agent focus
        focus = self.get_token_index_from_focus(expression, bucket_focus)

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
            msg = "Invalid move selected ({}) for expression({}) with focus({}). Rule({}) does not apply."
            raise Exception(msg.format(action, expression, token, type(operation)))

        change = operation.applyTo(token.rootClone())
        root = change.result.getRoot()
        out_problem = str(root)
        if not searching and self.verbose:
            output = """{:<25}: {}""".format(
                change.rule.name[:25], change.result.getRoot()
            )
            print("[{}] {}".format(str(agent.moves_remaining).zfill(2), output))
        out_env = env_state.encode_player(out_problem, agent.moves_remaining - 1)
        reward = self.get_state_value(out_env, searching)
        is_done = is_terminal_reward(reward) or agent.moves_remaining <= 0
        return out_env, reward, is_done

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

    def get_state_value(self, env_state: MathEnvironmentState, searching=False):
        """Get the value of the current state


        Input:
            env_state:     current env_state
            searching: boolean that is True when called by MCTS simulation

        Returns:
            r: the scalar reward value for the given environment state
               
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)

        # The agent is penalized for returning to a previous state.
        for key, group in groupby(sorted(agent.history)):
            list_group = list(group)
            list_count = len(list_group)
            if list_count <= 1:
                continue
            if not searching and self.verbose:
                print("\n[Penalty] re-entered previous state: {}".format(list_group[0]))
            return REWARD_PREVIOUS_LOCATION

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
                return REWARD_WIN

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if agent.moves_remaining <= 0:
            if not searching:
                self.write_draw(
                    "[Failed] exhausted moves:\n\t input: {}\n\t 1: {}\n".format(
                        self.expression_str, expression
                    )
                )
            return REWARD_LOSE

        # The game continues at a reward cost of one per step
        return REWARD_TIMESTEP
