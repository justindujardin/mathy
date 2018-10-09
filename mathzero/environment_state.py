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
from .core.profiler import profile_start, profile_end

# NOTE: store game mode as a bit on the data, then use different win conditions for each game.
#       hopefully the RL algorithm should be able to learn to play multiple math "games" with the
#       same model. Imagine:  train for 2 hours to get combining terms skill with simple case like "2x + 4x"
#                             then train for 4 hours on expanding that skill out to expressions with more
#                             like terms or terms with different variables.
MODE_ARITHMETIC = 0
MODE_SIMPLIFY = 1
MODE_COMBINE_LIKE_TERMS = 2


PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
GAME_MODE_OFFSET = 2


class MathAgentState(object):
    def __init__(self, player: int, move_count: int, problem: str, history=[]):
        self.player = player
        self.move_count = move_count
        self.problem = problem
        self.history = history
        if player != 1 and player != -1:
            raise ValueError("player must be 1 or -1, not: {}".format(player))

    @classmethod
    def copy(cls, from_state):
        return MathAgentState(
            from_state.player, from_state.move_count, from_state.problem
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
        width: int = 128,
        parser: ExpressionParser = None,
    ):
        self.parser = parser if parser is not None else ExpressionParser()
        self.width = width
        if problem is not None:
            self.agent_one = MathAgentState(1, 0, problem)
            self.agent_two = MathAgentState(-1, 0, problem)
        elif state is not None:
            self.width = state.width
            self.agent_one = MathAgentState.copy(state.agent_one)
            self.agent_two = MathAgentState.copy(state.agent_two)
        else:
            raise ValueError("either state or a problem must be provided")

    @classmethod
    def copy(cls, from_state):
        return MathEnvironmentState(state=from_state)

    def get_player(self, player: int) -> MathAgentState:
        """Helper to get a player by its index"""
        if player != 1 and player != -1:
            raise ValueError("player must be 1 or -1, not: {}".format(player))
        return self.agent_one if player == 1 else self.agent_two

    def encode_player(self, player: int, problem: str, move_count: int):
        """Encode a player's state into the env_state, and return the env_state"""
        out_state = MathEnvironmentState.copy(self)
        agent = out_state.get_player(player)
        agent.problem = problem
        agent.move_count = move_count
        return out_state

    def get_canonical_board(self, player: int):
        # print("gcb: {}".format(env_state.shape))
        out_state = MathEnvironmentState.copy(self)
        # We swap the agents if the player index doesn't match agent_one
        if player != 1:
            a = out_state.agent_two
            out_state.agent_two = out_state.agent_one
            out_state.agent_one = a
        return out_state

    def to_numpy(self):
        # We store 4 columns with length {self.width} each
        # The columns alternate content between the two players:
        #  data[0] == player_1 metadata
        #  data[1] == player_1 env_state
        #  data[2] == player_-1 metadata
        #  data[3] == player_-1 env_state
        # We store the data this way to
        data = numpy.zeros((4, self.width), dtype="float32")
        data = self.write_agent(0, data, self.agent_one, self.width)
        data = self.write_agent(2, data, self.agent_two, self.width)
        return data

    def write_agent(self, index, data, agent: MathAgentState, width: int):
        data[index][PLAYER_ID_OFFSET] = agent.player
        data[index][MOVE_COUNT_OFFSET] = agent.move_count
        # TODO: Decide how to detect/encode modes...
        data[index][GAME_MODE_OFFSET] = MODE_COMBINE_LIKE_TERMS
        features = self.parser.make_features(agent.problem)
        features = numpy.array(features, dtype="float32").flatten()
        features_len = len(features)
        for i in range(width):
            ch = features[i] if i < features_len else 0.0
            data[index + 1][i] = ch
        return data
