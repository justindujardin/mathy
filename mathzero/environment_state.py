import random
import numpy
from .core.tree import STOP
from .core.parser import ExpressionParser
from .training.problems import (
    MODE_ARITHMETIC,
    MODE_SIMPLIFY_POLYNOMIAL,
    MODE_SOLVE_FOR_VARIABLE,
)
from .model.features import (
    FEATURE_COLUMNS,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_MOVE_COUNT,
    FEATURE_TOKEN_TYPES,
    FEATURE_TOKEN_VALUES,
)

PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
GAME_MODE_OFFSET = 2

MODEL_WIDTH = 128
MODEL_HISTORY_LENGTH = 6


class MathAgentState(object):
    def __init__(self, move_count: int, problem: str, problem_type: int, history=None):
        self.move_count = move_count
        self.problem = problem
        self.problem_type = problem_type
        self.history = history[:] if history is not None else []

    @classmethod
    def copy(cls, from_state):
        return MathAgentState(
            from_state.move_count,
            from_state.problem,
            from_state.problem_type,
            from_state.history,
        )


class MathEnvironmentState(object):
    """Class for interacting with an Environment state.
    
    Mutating operations all return a copy of the environment adapter
    with its own state.

    This allocation strategy requires more memory but removes a class
    of potential issues around unintentional sharing of data and mutation
    by two different sources.
    """

    def __init__(
        self,
        state=None,
        problem: str = None,
        problem_type: int = MODE_SIMPLIFY_POLYNOMIAL,
        width: int = MODEL_WIDTH,
        parser: ExpressionParser = None,
        history_length: int = MODEL_HISTORY_LENGTH,
    ):
        self.parser = parser if parser is not None else ExpressionParser()
        self.width = width
        self.history_length = history_length
        if problem is not None:
            self.agent = MathAgentState(0, problem, problem_type)
        elif state is not None:
            self.width = state.width
            self.agent = MathAgentState.copy(state.agent)
        else:
            raise ValueError("either state or a problem must be provided")

    @classmethod
    def copy(cls, from_state):
        return MathEnvironmentState(state=from_state)

    def clone(self):
        return MathEnvironmentState(state=self)

    def encode_player(self, problem: str, move_count: int):
        """Encode a player's state into the env_state, and return the env_state"""
        out_state = MathEnvironmentState.copy(self)
        agent = out_state.agent
        agent.history.append(problem)
        agent.problem = problem
        agent.move_count = move_count
        return out_state

    def make_features(self, tokens_or_text):
        """
        Make a list of tokenized features from text input, to make the learning 
        objective easier for the model.
        """
        # Token list
        tokens = None
        if type(tokens_or_text) == list:
            tokens = tokens_or_text
        elif type(tokens_or_text) == str:
            tokens = self.parser.tokenize(tokens_or_text)
        else:
            raise ValueError(
                "features can only be created from a token list or str input expression"
            )

    def to_input_features(self):
        types = []
        values = []

        def add_tokens(tokens):
            nonlocal types, values
            for t in tokens:
                types.append(t.type)
                values.append(t.value)
            # Insert break characters
            types.append(-1)
            values.append("|")

        for problem in self.agent.history:
            add_tokens(self.parser.tokenize(problem))
        add_tokens(self.parser.tokenize(self.agent.problem))
        input_features = {
            FEATURE_MOVE_COUNT: self.agent.move_count,
            FEATURE_NODE_COUNT: len(values),
            FEATURE_PROBLEM_TYPE: self.agent.problem_type,
            FEATURE_TOKEN_TYPES: types,
            FEATURE_TOKEN_VALUES: values,
        }
        return input_features

