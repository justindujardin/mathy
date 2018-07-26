import random
import numpy
from .math.expressions import (
    MathExpression,
    ConstantExpression,
    STOP,
    AddExpression,
    VariableExpression,
)
from .math.parser import ExpressionParser
from .math.util import termsAreLike
from .math.rules import (
    BaseRule,
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)
from .math.profiler import profile_start, profile_end


class MathBoard:
    def __init__(self, width):
        self.width = width

    def encode_board(self, text):
        # We store 4 columns with length 256 each
        # The columns alternate content between the two players:
        #  data[0] == player_1 metadata
        #  data[1] == player_1 board
        #  data[2] == player_-1 metadata
        #  data[3] == player_-1 board
        data = numpy.zeros((4, self.width), dtype="float32")
        # Encode the text twice, once for each player.
        text_len = len(text)
        data[0][0] = 1
        data[2][0] = -1
        for i in range(self.width):
            ch = text[i] if i < text_len else " "
            data[1][i] = data[3][i] = ord(ch)

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

    def encode_player(self, board, player, text, move_count, focus_index=0):
        """Encode a player's state into the board, and return the board"""
        # print("ep: {}".format(board.shape))
        data = numpy.zeros((2, self.width), dtype="float32")
        text = str(text)  # ensure if given an expression it is cast to a string
        other_data = self.slice_player_data(board, player * -1)
        data[0][0] = player
        data[0][1] = move_count
        data[0][2] = focus_index
        text_len = len(text)
        for i in range(self.width):
            ch = text[i] if i < text_len else " "
            data[1][i] = ord(ch)
        # print("encode move_count i: {}".format(data[0][0]))
        # print("encode focus is : {}".format(focus_index))
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
        player_index = int(player_data[0][0])
        move_count = int(player_data[0][1])
        focus_index = int(player_data[0][2])
        # print("{}] decoded player is : {}".format(player, player_index))
        # print("{}] decoded move is : {}".format(player, move_count))
        # print("{}] decoded focus is : {}".format(player, focus_index))
        text = "".join([chr(p) for p in player_data[1]]).strip()
        # print("{}] text is : {}".format(player, text))
        return text, move_count, focus_index, player_index

    def get_canonical_board(self, board, player):
        # print("gcb: {}".format(board.shape))
        result = board.copy()
        split = numpy.vsplit(result, 2)
        if player == 1:
            result = numpy.vstack((split[0], split[1]))
        elif player == -1:
            result = numpy.vstack((split[1], split[0]))
        return result

