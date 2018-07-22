"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
"""


class MathBoard:
    def __init__(self, width, history_length):
        "Set up initial board configuration."
        self.width = width
        self.history_length = history_length

        # Create the empty board array.
        self.pieces = [None] * self.width
        for i in range(self.width):
            self.pieces[i] = [0] * self.history_length

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.
        return list(moves)

    def has_legal_moves(self, color):
        return False

    def get_moves_for_square(self, square):
        return None

    def execute_move(self, move, color):
        print("excute move: {}, {}".format(move, color))
        pass
