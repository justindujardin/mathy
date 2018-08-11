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


class EnvironmentState:
    def __init__(self, width):
        self.width = width
        self.parser = ExpressionParser()

    def encode_board(self, text):
        # We store 4 columns with length {self.width} each
        # The columns alternate content between the two players:
        #  data[0] == player_1 metadata
        #  data[1] == player_1 board
        #  data[2] == player_-1 metadata
        #  data[3] == player_-1 board
        # We store the data this way to
        mode = MODE_COMBINE_LIKE_TERMS
        data = numpy.zeros((4, self.width), dtype="float32")
        data[0][PLAYER_ID_OFFSET] = 1
        data[0][MOVE_COUNT_OFFSET] = 0
        data[0][GAME_MODE_OFFSET] = mode
        data[2][PLAYER_ID_OFFSET] = -1
        data[2][MOVE_COUNT_OFFSET] = 0
        data[2][GAME_MODE_OFFSET] = mode
        features = self.parser.make_features(text)
        features = numpy.array(features, dtype="float32").flatten()
        features_len = len(features)
        for i in range(self.width):
            ch = features[i] if i < features_len else 0.0
            # Each player starts with the same encoded text
            data[1][i] = data[3][i] = ch

        return data

    def slice_player_data(self, board, player):
        # print("spd: {}".format(board.shape))
        if board is None:
            raise Exception("there is no board to decode player from")
        shaped = numpy.vsplit(board.copy(), 2)
        if player is 1:
            player_data = shaped[0]
        elif player is -1:
            player_data = shaped[1]
        else:
            raise ValueError(
                "invalid player, must be 1 or -1 but is: {}".format(player)
            )
        return player_data

    def encode_player(self, board, player, features, move_count):
        """Encode a player's state into the board, and return the board"""
        features_len = len(features)
        features = numpy.array(features, dtype="float32").flatten()
        # print("ep: {}".format(board.shape))
        data = numpy.zeros((2, self.width), dtype="float32")
        other_data = self.slice_player_data(board, player * -1)
        data[0][PLAYER_ID_OFFSET] = player
        data[0][MOVE_COUNT_OFFSET] = move_count
        features_len = len(features)
        for i in range(self.width):
            ch = features[i] if i < features_len else 0.0
            data[1][i] = ch
        # Order the data based on which player the move is for.
        if player == 1:
            result = numpy.vstack((data, other_data))
        else:
            result = numpy.vstack((other_data, data))
        # print("ep: {} (p{} return value)".format(result.shape, player))
        return result

    def decode_player(self, board, player):
        # print("dp: {}".format(board.shape))
        player_data = self.slice_player_data(board, player)
        player_index = int(player_data[0][PLAYER_ID_OFFSET])
        move_count = int(player_data[0][MOVE_COUNT_OFFSET])
        # print("{}] decoded player is : {}".format(player, player_index))
        # print("{}] decoded move is : {}".format(player, move_count))
        out_features = []
        for i in range(0, self.width, 2):
            value = player_data[1][i]
            type = int(player_data[1][i + 1])
            out_features.append([value, type])
            if type == TokenEOF:
                break

        # print("{}] text is : {}".format(player, text))
        return out_features, move_count, player_index

    def get_canonical_board(self, board, player):
        # print("gcb: {}".format(board.shape))
        result = board.copy()
        split = numpy.vsplit(result, 2)
        if player == 1:
            result = numpy.vstack((split[0], split[1]))
        elif player == -1:
            result = numpy.vstack((split[1], split[0]))
        return result
