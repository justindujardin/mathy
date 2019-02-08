import random
import numpy
from .core.expressions import (
    MathExpression,
    ConstantExpression,
    STOP,
    AddExpression,
    VariableExpression,
)
from .core.parser import ExpressionParser
from .core.tokenizer import Token, TokenEOF
from .core.util import termsAreLike
from .core.rules import (
    BaseRule,
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)
from .core.problems import (
    MODE_ARITHMETIC,
    MODE_SIMPLIFY_POLYNOMIAL,
    MODE_SOLVE_FOR_VARIABLE,
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

    def to_numpy(self):
        data = numpy.zeros((self.history_length + 1, self.width), dtype="float32")
        # TODO: Decide how to detect/encode modes...
        data[0][0] = self.agent.problem_type
        data = self.write_problem(1, data, self.agent.problem)
        # Encode up to last (history_length-1) moves into the agent "memory"
        history = self.agent.history[-(self.history_length - 1) :]
        for i, problem in enumerate(history):
            data = self.write_problem(i + 2, data, problem)
        return data

    def write_agent(self, index, data, agent: MathAgentState, width: int):
        features = self.parser.make_features(agent.problem)
        features = numpy.array(features, dtype="float32").flatten()
        features_len = len(features)
        for i in range(width):
            ch = features[i] if i < features_len else 0.0
            data[index + 1][i] = ch
        return data

    def write_problem(self, index, data, problem: str):
        features = self.parser.make_features(problem)
        features = numpy.array(features, dtype="float32").flatten()
        features_len = len(features)
        for i in range(self.width):
            ch = features[i] if i < features_len else 0.0
            data[index][i] = ch
        return data
