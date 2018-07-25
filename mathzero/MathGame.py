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


class MetaAction:
    def count_nodes(self, expression: MathExpression) -> int:
        count = 0
        found = -1

        def visit_fn(node, depth, data):
            nonlocal count, found
            if node.id == expression.id:
                found = count
            count = count + 1

        expression.getRoot().visitInorder(visit_fn)
        return count, found


class VisitBeforeAction(MetaAction):
    def canVisit(self, expression: MathExpression) -> bool:
        count, found = self.count_nodes(expression)
        return bool(found > 1 and count > 1)

    def visit(self, focus_index: int):
        return focus_index - 1


class VisitAfterAction(MetaAction):
    def canVisit(self, expression: MathExpression) -> bool:
        count, found = self.count_nodes(expression)
        return bool(found >= 0 and found < count - 2)

    def visit(self, focus_index: int):
        return focus_index + 1


class MathGame:
    """
    Implement a math solving game where players have two distinct win conditions
    that require different strategies for solving. The first win-condition is for 
    the player to execute the right sequence of actions to reduce a math expression
    to its most basic representation. The second is for the other player to expand 
    the simple representations into more verbose expressions.

    Ideally a fully-trained player will be able to both simplify arbitrary mathematical
    expressions, and expand upon arbitrary parts of expressions to generate more complex
    representations that can be used to expand on concepts that a user may struggle with.
    """

    tokens = list(" abcdefghijklmnopqrstuvwxyz01234567890.!=()^*+-/")

    def __init__(self, expression_str: str):
        self.width = 32
        self.parser = ExpressionParser()
        self.expression_str = str(expression_str)
        if len(list(self.expression_str)) > self.width:
            raise Exception(
                'Expression "{}" is too long for the current model to process. Max width is: {}'.format(
                    self.expression_str, self.width
                )
            )
        self.input_characters = MathGame.tokens
        self.tokens_count = len(self.input_characters)
        # use whitespace for padding
        self.padding_token = self.input_characters[0]
        self.token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)]
        )
        self.input_data = self.encode_board(self.expression_str, 0)
        self.available_actions = [VisitAfterAction(), VisitBeforeAction()]
        self.available_rules = [
            CommutativeSwapRule(),
            # DistributiveFactorOutRule(),
            # DistributiveMultiplyRule(),
            AssociativeSwapRule(),
            ConstantsSimplifyRule(),
        ]

    def getInitBoard(self):
        """return a numpy encoded version of the input expression"""
        return self.input_data.copy()
        # NOTE: This is called for each episode, so it can be thought of like "onInitEpisode()"
        # return self.encode_board(self.expression_str, 0)

    def getBoardSize(self):
        """return shape (x,y) of board dimensions"""
        return (self.width + 1, self.tokens_count)

    def getActionSize(self):
        """Return number of all possible actions"""
        return len(self.available_rules) + len(self.available_actions)

    def getNextState(self, board, player, action, searching=False):
        """
        Input:
            board:     current board
            player:    current player (1 or -1)
            action:    action taken by current player
            searching: boolean set to True when called by MCTS

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        
        text, move_count, focus_index = self.decode_board(board)
        expression = self.parser.parse(text)
        token = self.getFocusToken(expression, focus_index)
        actions = self.available_actions + self.available_rules
        operation = actions[action]
        debug = True

        if isinstance(operation, BaseRule) and operation.canApplyTo(token):
            change = operation.applyTo(token.rootClone())
            root = change.end.getRoot()
            if not searching and debug:
                print(
                    "[{}][out={}] {}".format(
                        move_count, change.end.id, change.describe()
                    )
                )
            out_board = self.encode_board(root, move_count + 1, focus_index)
        elif isinstance(operation, MetaAction):
            if not searching and debug:
                direction = (
                    "behind" if isinstance(operation, VisitBeforeAction) else "ahead"
                )
                print(
                    "[{}] 👀 Looking {} at: {}".format(
                        move_count,
                        direction,
                        self.getFocusToken(expression, operation.visit(focus_index)),
                    )
                )
            # DONE BELOW: actions for setting the currently focused node. This encourages interpretibility
            # by forcing the action stream to include where the machine is transitioning its focus
            # to before taking actions. It also reduces the potential number of valid actions at a given
            # point to a constant number rather than the number of potential actions across the entire
            # expression, which could be tens or hundreds instead of a handful.
            out_board = self.encode_board(
                expression, move_count + 1, operation.visit(focus_index)
            )
            # profile_end('getNextState.applyMetaAction')
        else:
            if not searching and debug:
                print(
                    "[{}] skip because invalid move was selected: {}, {}".format(
                        move_count, action, operation
                    )
                )
            out_board = self.encode_board(expression, move_count + 1, focus_index)
        return out_board, player

    def getFocusToken(
        self, expression: MathExpression, focus_index: int
    ) -> MathExpression:
        """Get the token that is `focus_index` from the left of the expression"""
        count = 0
        result = None

        def visit_fn(node, depth, data):
            nonlocal result, count
            result = node
            if count == focus_index:
                return STOP
            count = count + 1

        expression.visitInorder(visit_fn)
        return result

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        expression_text, _, focus_index = self.decode_board(board)
        expression = self.parser.parse(expression_text)
        token = self.getFocusToken(expression, focus_index)
        actions = [0] * self.getActionSize()
        count = 0
        # Meta actions first
        for action in self.available_actions:
            if action.canVisit(token):
                actions[count] = 1
            count = count + 1
        # Rules of numbers
        for rule in self.available_rules:
            if rule.canApplyTo(token):
                actions[count] = 1
            count = count + 1

        # print_list = self.available_actions + self.available_rules
        # [
        #     print("action[{}][{}] = {}".format(i, bool(a), type(print_list[i]) if a != 0 else ''))
        #     for i, a in enumerate(actions)
        # ]
        return actions

    def getGameEnded(self, board, player, searching=False):
        """
        Input:
            board:     current board
            player:    current player (1 or -1)
            searching: boolean that is True when called by MCTS simulation

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        expression_text, move_count, _ = self.decode_board(board)
        expression = self.parser.parse(expression_text)
        if move_count > 20:
            if not searching:
                print(
                    "[LOSE] ENDED WITH: {} => {}!".format(
                        self.expression_str, expression
                    )
                )
            else:
                print(".")
            return -1

        # It's over if the expression is reduced to a single constant
        if (
            isinstance(expression, ConstantExpression)
            and len(expression.getChildren()) == 0
        ):

            # eval = self.parser.parse(self.expression_str).evaluate()
            # found = expression.evaluate()
            # # TODO: This int cast is a cheap hack to not worry about epsilon differences
            # # TODO: Remove this when working with lots of floating point stuff
            # if int(eval) != int(found):
            #     if not searching:
            #         print(
            #             "[LOSE] ERROR: reduced '{}' to constant, but evals differ. Expected '{}' but got '{}'!".format(
            #                 self.expression_str, eval, found
            #             )
            #         )
            #     return -1

            # Holy shit it won!
            if not searching:
                print("[WIN] {} => {}!".format(self.expression_str, expression))
            else:
                print(".")

            return 1
        # Check for simplification down to a single addition with constant/variable
        if isinstance(expression, AddExpression):
            constant = None
            variable = None

            if isinstance(expression.left, ConstantExpression):
                constant = expression.left
            elif isinstance(expression.right, ConstantExpression):
                constant = expression.right
            else:
                return 0

            if isinstance(expression.right, VariableExpression):
                constant = expression.right
            elif isinstance(expression.right, VariableExpression):
                constant = expression.right
            else:
                return 0

            # TODO: Compare constant/variable findings to verify their accuracy.
            #       This helps until we're 100% confident in the parser/serializer consistency
            #
            # Holy shit it won!
            if not searching:
                print("[TERMWIN] {} => {}!".format(self.expression_str, expression))
                print("TERM WIN WITH CONSTANT: {}".format(constant))
                print("TERM WIN WITH VARIABLE: {}".format(variable))
            else:
                print(".")
            return 1

        # The game continues
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """conversion of board to a string format, required by MCTS for hashing."""
        expression_text, move_count, focus_index = self.decode_board(board)
        return "f{}_m{}_e{}".format(focus_index, move_count, expression_text)

    def encode_board(self, text, move_count, focus_index=0):
        """Encode the given math expression string into tokens on a game board"""
        data = numpy.zeros((self.width + 1, self.tokens_count), dtype="float32")
        characters = list(str(text))
        # print("encode move as: {}".format(move_count))
        # Store move count in first column/row
        data[0][0] = move_count
        data[0][1] = focus_index
        for i, ch in enumerate(characters):
            data[i + 1][self.token_index[ch]] = 1.
        # print("encode move_count i: {}".format(data[0][0]))
        # print("encode focus is : {}".format(focus_index))
        return data

    def decode_board(self, board):
        """Decode the given board into an expression string"""
        token_indices = numpy.argmax(board, axis=1)
        move_count = int(board[0][0])
        focus_index = int(board[0][1])
        # print("decoded move is : {}".format(move_count))
        # print("decoded focus is : {}".format(focus_index))
        text = [self.tokens[t] for i, t in enumerate(token_indices) if i != 0]
        return "".join(text).strip(), move_count, focus_index
