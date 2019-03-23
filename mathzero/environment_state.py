import random
import numpy
from .core.tree import STOP
from .core.parser import ExpressionParser
from .core.tokenizer import TokenEOF
from .training.problems import (
    MODE_ARITHMETIC,
    MODE_SIMPLIFY_POLYNOMIAL,
    MODE_SOLVE_FOR_VARIABLE,
)

from .model.features import (
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_TOKEN_TYPES,
    FEATURE_TOKEN_VALUES,
    FEATURE_LAST_TOKEN_TYPES,
    FEATURE_LAST_TOKEN_VALUES,
    pad_array,
)

PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
GAME_MODE_OFFSET = 2

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"


class MathAgentState(object):
    def __init__(
        self,
        moves_remaining: int,
        problem: str,
        problem_type: int,
        reward=0.0,
        history=None,
    ):
        self.moves_remaining = moves_remaining
        self.problem = problem
        self.reward = reward
        self.problem_type = problem_type
        self.history = history[:] if history is not None else []

    @classmethod
    def copy(cls, from_state):
        return MathAgentState(
            from_state.moves_remaining,
            from_state.problem,
            from_state.reward,
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
        max_moves=10,
        problem_type: int = MODE_SIMPLIFY_POLYNOMIAL,
    ):
        self.parser = ExpressionParser()
        self.max_moves = max_moves
        if problem is not None:
            self.agent = MathAgentState(max_moves, problem, problem_type)
        elif state is not None:
            self.max_moves = state.max_moves
            self.agent = MathAgentState.copy(state.agent)
        else:
            raise ValueError("either state or a problem must be provided")

    @classmethod
    def copy(cls, from_state):
        return MathEnvironmentState(state=from_state)

    def clone(self):
        return MathEnvironmentState(state=self)

    def encode_player(self, problem: str, moves_remaining: int):
        """Encode a player's state into the env_state, and return the env_state"""
        out_state = MathEnvironmentState.copy(self)
        agent = out_state.agent
        agent.history.append(problem)
        agent.problem = problem
        agent.moves_remaining = moves_remaining
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

    def to_input_features(self, return_batch=False):
        """Output a one element array of features that can be fed to the 
        neural network for prediction.

        When return_batch is true, the outputs will have an array of features for each
        so the result can be directly passed to the predictor.
        """
        import tensorflow as tf

        expression = self.parser.parse(self.agent.problem)
        tokens = self.parser.tokenize(self.agent.problem)
        types = []
        values = []
        for t in tokens:
            types.append(t.type)
            values.append(t.value)

        # Generate previous state token features because we pass observations
        # out of order as single entries, rather than as a sequence for the episode.
        last_types = []
        last_values = []
        max_len = len(types)
        if len(self.agent.history) > 0:
            last_problem = self.agent.history[0]
            expression = self.parser.parse(last_problem)
            tokens = self.parser.tokenize(last_problem)
            for t in tokens:
                last_types.append(t.type)
                last_values.append(t.value)
            if len(last_types) > max_len:
                max_len = len(last_types)

        # Pad all the sequence inputs to the same length
        types = pad_array(types, max_len, TokenEOF)
        values = pad_array(values, max_len, " ")
        last_types = pad_array(last_types, max_len, TokenEOF)
        last_values = pad_array(last_values, max_len, " ")

        def maybe_wrap(value):
            nonlocal return_batch
            return [value] if return_batch else value

        return {
            FEATURE_TOKEN_TYPES: maybe_wrap(types),
            FEATURE_TOKEN_VALUES: maybe_wrap(values),
            FEATURE_LAST_TOKEN_TYPES: maybe_wrap(last_types),
            FEATURE_LAST_TOKEN_VALUES: maybe_wrap(last_values),
            FEATURE_NODE_COUNT: maybe_wrap(len(expression.toList())),
            FEATURE_MOVE_COUNTER: maybe_wrap(
                int(self.max_moves - self.agent.moves_remaining)
            ),
            FEATURE_MOVES_REMAINING: maybe_wrap(self.agent.moves_remaining),
            FEATURE_PROBLEM_TYPE: maybe_wrap(int(self.agent.problem_type)),
        }


def to_sparse_tensor(sequence, dtype=None, fill_value=-1):
    import tensorflow as tf

    if dtype is None:
        dtype = tf.int32

    tensor = tf.convert_to_tensor(sequence, dtype=dtype)
    idx = tf.where(tf.not_equal(tensor, fill_value))
    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    sparse = tf.SparseTensor(
        idx, tf.gather_nd(tensor, idx), tf.shape(tensor, out_type=dtype)
    )
    return sparse
