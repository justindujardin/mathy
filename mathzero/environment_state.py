import random
import numpy
from collections import namedtuple
from .core.tree import STOP
from .core.expressions import MathExpression, MathTypeKeys
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
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    pad_array,
)
import json

PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
GAME_MODE_OFFSET = 2

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"


# Capture summarized environment state for a previous timestep so the agent can use
# context from its history when making new predictions.
AgentTimeStep = namedtuple("AgentTimeStep", ["raw", "focus", "action"])

# Build a context sensitive vector for each token
class ContextualToken(namedtuple("ContextualToken", ["previous", "current", "next"])):
    def __str__(self):
        return f"prev:{previous}, current:{current}, next:{next}"


# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE: Need to switch from token input features to node input features
# NOTE: so they have a 1:1 relationship with focus index
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:
# NOTE:


class MathAgentState(object):
    def __init__(
        self,
        moves_remaining: int,
        problem: str,
        problem_type: int,
        reward=0.0,
        history=None,
        focus_index=0,
        last_action=None,
    ):
        self.moves_remaining = moves_remaining
        self.focus_index = focus_index
        self.problem = problem
        self.last_action = last_action
        self.reward = reward
        self.problem_type = problem_type
        self.history = history[:] if history is not None else []

    @classmethod
    def copy(cls, from_state):
        return MathAgentState(
            moves_remaining=from_state.moves_remaining,
            problem=from_state.problem,
            reward=from_state.reward,
            problem_type=from_state.problem_type,
            history=from_state.history,
            focus_index=from_state.focus_index,
            last_action=from_state.last_action,
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

    def encode_player(
        self, problem: str, action: int, focus_index: int, moves_remaining: int
    ):
        """Encode a player's state into the env_state, and return the env_state"""
        out_state = MathEnvironmentState.copy(self)
        agent = out_state.agent
        agent.history.append(AgentTimeStep(problem, focus_index, action))
        agent.problem = problem
        agent.action = action
        agent.moves_remaining = moves_remaining
        agent.focus_index = focus_index
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

    def get_node_vectors(self, expression: MathExpression):
        """Get a set of context-sensitive vectors for a given expression"""
        # pre-order is important to match up with how the Meta focus actions visit the tree
        # NOTE: see agent_actions or MetaAction
        nodes = expression.toList("preorder")
        vectors = []
        nodes_len = len(nodes)
        node_masks = []
        pad_value = MathTypeKeys["empty"]
        for i, t in enumerate(nodes):
            last = pad_value if i == 0 else nodes[i - 1].type_id
            next = pad_value if i > nodes_len - 2 else nodes[i + 1].type_id
            vectors.append((last, t.type_id, next))
        return vectors

    def to_input_features(self, return_batch=False):
        """Output a one element array of features that can be fed to the 
        neural network for prediction.

        When return_batch is true, the outputs will have an array of features for each
        so the result can be directly passed to the predictor.
        """
        import tensorflow as tf

        expression = self.parser.parse(self.agent.problem)
        # Padding value is a result tuple with empty values for prev/current/next
        pad_value = (
            MathTypeKeys["empty"],
            MathTypeKeys["empty"],
            MathTypeKeys["empty"],
        )

        # Generate context vectors for the current state's expression tree
        vectors = self.get_node_vectors(expression)

        # Provide context vectors for the previous state if there is one
        last_vectors = [pad_value] * len(vectors)
        if len(self.agent.history) > 1:
            last_expression = self.parser.parse(self.agent.history[-2].raw)
            last_vectors = self.get_node_vectors(last_expression)

            # If the sequences differ in length, pad to the longest one
            if len(last_vectors) != len(vectors):
                max_len = max(len(last_vectors), len(vectors))
                last_vectors = pad_array(last_vectors, max_len, pad_value)
                vectors = pad_array(vectors, max_len, pad_value)

        # After padding is done, generate reversed vectors for the bilstm
        vectors_reversed = vectors[:]
        vectors_reversed.reverse()

        last_vectors_reversed = last_vectors[:]
        last_vectors_reversed.reverse()

        def maybe_wrap(value):
            nonlocal return_batch
            return [value] if return_batch else value

        return {
            FEATURE_FWD_VECTORS: maybe_wrap(vectors),
            FEATURE_BWD_VECTORS: maybe_wrap(vectors_reversed),
            FEATURE_LAST_FWD_VECTORS: maybe_wrap(last_vectors),
            FEATURE_LAST_BWD_VECTORS: maybe_wrap(last_vectors_reversed),
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
