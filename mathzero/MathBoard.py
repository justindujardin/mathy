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

    # Token 0 is whitespace so numpy.zeros is just a big whitespace input
    tokens = list(" abcdefghijklmnopqrstuvwxyz01234567890.!=()^*+-/")

    def __init__(self, width):
        self.width = width
        self.input_characters = MathBoard.tokens
        self.tokens_count = len(self.input_characters)
        self.token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)]
        )

    def encode_board(self, text):
        # width size input for each player + 1 to store player specific state
        combined_width = self.width * 2 + 2
        data = numpy.zeros((combined_width, self.tokens_count), dtype="float32")
        characters = list(str(text))
        p1_offset = 1
        p2_offset = self.width + 2
        # Encode the text twice, once for each player.
        for i, ch in enumerate(characters):
            data[i + p1_offset][self.token_index[ch]] = 1.
            data[i + p2_offset][self.token_index[ch]] = 1.

        return data

    def slice_player_data(self, board, player):
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
        player_data = numpy.zeros((self.width + 1, self.tokens_count))
        other_data = self.slice_player_data(board, player * -1)
        characters = list(str(text))
        # print("encode move as: {}".format(move_count))
        # Store move count in first column/row
        player_data[0][0] = move_count
        player_data[0][1] = focus_index
        for i, ch in enumerate(characters):
            player_data[i + 1][self.token_index[ch]] = 1.
        # print("encode move_count i: {}".format(data[0][0]))
        # print("encode focus is : {}".format(focus_index))

        # Order the data based on which player the move is for.
        if player == 1:
            return numpy.vstack((player_data, other_data))
        else:
            return numpy.vstack((other_data, player_data))
        return player_data

    def decode_player(self, board, player):
        player_data = self.slice_player_data(board, player)
        token_indices = numpy.argmax(player_data, axis=1)
        move_count = int(player_data[0][0])
        focus_index = int(player_data[0][1])
        # print("{}] decoded move is : {}".format(player, move_count))
        # print("{}] decoded focus is : {}".format(player, focus_index))
        text = "".join(
            [self.tokens[t] for i, t in enumerate(token_indices) if i != 0]
        ).strip()
        # print("{}] text is : {}".format(player, text))
        return text, move_count, focus_index

    def get_canonical_board(self, board, player):
        result = board.copy()
        if player != 1:
            split = numpy.vsplit(board, 2)
            result = numpy.vstack((split[0], split[1]))
        return result

