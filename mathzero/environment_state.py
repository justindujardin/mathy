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


PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
FOCUS_INDEX_OFFSET = 2
META_COUNTER_OFFSET = 3
LAST_ACTION_OFFSET = 4

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

        data = numpy.zeros((4, self.width), dtype="float32")
        data[0][PLAYER_ID_OFFSET] = 1
        data[0][MOVE_COUNT_OFFSET] = 0
        data[0][FOCUS_INDEX_OFFSET] = 1 # Start focus at 1 to bias away from leaf nodes that cannot be operators. 
        data[0][META_COUNTER_OFFSET] = 0
        data[0][LAST_ACTION_OFFSET] = -1

        data[2][PLAYER_ID_OFFSET] = -1
        data[2][MOVE_COUNT_OFFSET] = 0
        data[2][FOCUS_INDEX_OFFSET] = 1
        data[2][META_COUNTER_OFFSET] = 0
        data[2][LAST_ACTION_OFFSET] = -1

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

    def encode_player(
        self,
        board,
        player,
        features,
        move_count,
        focus_index,
        meta_counter,
        last_action,
    ):
        """Encode a player's state into the board, and return the board"""
        features_len = len(features)
        features = numpy.array(features, dtype="float32").flatten()
        # print("ep: {}".format(board.shape))
        data = numpy.zeros((2, self.width), dtype="float32")
        other_data = self.slice_player_data(board, player * -1)
        data[0][PLAYER_ID_OFFSET] = player
        data[0][MOVE_COUNT_OFFSET] = move_count
        data[0][FOCUS_INDEX_OFFSET] = focus_index
        data[0][META_COUNTER_OFFSET] = meta_counter
        data[0][LAST_ACTION_OFFSET] = last_action
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
        focus_index = int(player_data[0][FOCUS_INDEX_OFFSET])
        meta_counter = int(player_data[0][META_COUNTER_OFFSET])
        last_action = int(player_data[0][LAST_ACTION_OFFSET])
        # print("{}] decoded player is : {}".format(player, player_index))
        # print("{}] decoded move is : {}".format(player, move_count))
        # print("{}] decoded focus is : {}".format(player, focus_index))
        out_features = []
        for i in range(0, self.width, 2):
            value = player_data[1][i]
            type = int(player_data[1][i + 1])
            out_features.append([value, type])
            if type == TokenEOF:
                break

        # print("{}] text is : {}".format(player, text))
        return out_features, move_count, focus_index, player_index, meta_counter, last_action

    def get_canonical_board(self, board, player):
        # print("gcb: {}".format(board.shape))
        result = board.copy()
        split = numpy.vsplit(result, 2)
        if player == 1:
            result = numpy.vstack((split[0], split[1]))
        elif player == -1:
            result = numpy.vstack((split[1], split[0]))
        return result
