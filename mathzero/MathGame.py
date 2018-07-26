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
from .math.util import termsAreLike, isAddSubtract
from .math.rules import (
    BaseRule,
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
)
from .math.profiler import profile_start, profile_end
from .MathBoard import MathBoard


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

    width = 32

    def __init__(self, expression_str: str):
        self.width = MathGame.width
        self.parser = ExpressionParser()
        self.expression_str = str(expression_str)
        if len(list(self.expression_str)) > self.width:
            raise Exception(
                'Expression "{}" is too long for the current model to process. Max width is: {}'.format(
                    self.expression_str, self.width
                )
            )
        self.input_data = MathBoard(self.width).encode_board(self.expression_str)
        self.available_actions = [VisitAfterAction(), VisitBeforeAction()]
        self.available_rules = [
            CommutativeSwapRule(),
            DistributiveFactorOutRule(),
            DistributiveMultiplyRule(),
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
        # 2 columns per player, the first for turn data, the second for text inputs
        return (4, self.width)

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
        b = MathBoard(self.width)
        text, move_count, focus_index, _ = b.decode_player(board, player)
        # print("gns: {}".format(player))
        expression = self.parser.parse(text)
        token = self.getFocusToken(expression, focus_index)
        actions = self.available_actions + self.available_rules
        operation = actions[action]
        debug = False
        # searching = False

        # Enforce constraints to keep training time and complexity down
        # - can't commutative swap immediately to return to previous state.
        # - can't focus on the same token twice without taking a valid other
        #   action inbetween

        if isinstance(operation, BaseRule) and operation.canApplyTo(token):
            change = operation.applyTo(token.rootClone())
            root = change.end.getRoot()
            if not searching and debug:
                print("[{}] {}".format(move_count, change.describe()))
            out_board = b.encode_player(
                board, player, root, move_count + 1, focus_index
            )
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
            out_board = b.encode_player(
                board, player, expression, move_count + 1, operation.visit(focus_index)
            )
            # profile_end('getNextState.applyMetaAction')
        else:
            raise Exception(
                "\n\nPlayer: {}\n\tExpression: {}\n\tFocus: {}\n\tIndex: {}\n\tinvalid move selected: {}, {}".format(
                    player, expression, token, focus_index, action, type(operation)
                )
            )
        return out_board, player * -1

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
        b = MathBoard(self.width)

        expression_text, _, focus_index, board_player = b.decode_player(board, player)
        # Valid moves depend on player, but this is always called with canonical board
        # which normalizes the player away. We add the player ID to MCTS hash keys to
        # work around it, and then normalize the player moves per hash key here by extracting
        # the origin player ID and normalizing the results to be for that user.
        if player != board_player:
            expression_text, _, focus_index, _ = b.decode_player(board, player * -1)
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
        #     print(
        #         "Player{} action[{}][{}] = {}".format(
        #             player, i, bool(a), type(print_list[i]) if a != 0 else ""
        #         )
        #     )
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
        b = MathBoard(self.width)
        expression_text, move_count, _, _ = b.decode_player(board, player)
        # print("Expression = {}".format(expression_text))
        expression = self.parser.parse(expression_text)

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
                print(
                    "\n[Player{}][WIN] {} => {}!".format(
                        player, self.expression_str, expression
                    )
                )
            else:
                pass
                # print(".")

            return 1
        # Check for simplification down to a single addition with constant/variable
        add_sub = (
            expression
            if isinstance(expression, AddExpression)
            else expression.findByType(AddExpression)
        )
        if add_sub and add_sub.parent is None:
            constant = None
            variable = None

            if isinstance(add_sub.left, ConstantExpression):
                constant = add_sub.left
            elif isinstance(add_sub.right, ConstantExpression):
                constant = add_sub.right

            if isinstance(add_sub.right, VariableExpression):
                constant = add_sub.right
            elif isinstance(add_sub.right, VariableExpression):
                constant = add_sub.right

            # TODO: Compare constant/variable findings to verify their accuracy.
            #       This helps until we're 100% confident in the parser/serializer consistency
            #
            # Holy shit it won!
            if not searching and constant and variable:
                print(
                    "\n[Player{}][TERMWIN] {} => {}!".format(
                        player, self.expression_str, expression
                    )
                )
                print("TERM WIN WITH CONSTANT: {}".format(constant))
                print("TERM WIN WITH VARIABLE: {}".format(variable))
                return 1

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, it should be honored.
        if move_count > 20:
            if not searching:
                print(
                    "\n[Player {}][LOSE] ENDED WITH: {} => {}!".format(
                        player, self.expression_str, expression
                    )
                )
            else:
                pass
                # print(".")
            return -1

        # The game continues
        return 0

    # ===================================================================
    #
    #       STOP! THERE IS NOTHING YOU WANT TO CHANGE BELOW HERE.
    #
    # ===================================================================

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
        # print("gcf: {}".format(player))
        return MathBoard(self.width).get_canonical_board(board, player)

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

    def getPolicyKey(self, board):
        """conversion of board to a string format, required by MCTS for hashing."""
        b = MathBoard(self.width)
        # This is always called for the canonical board which means the
        # current player is always in player1 slot:
        e1, m1, f1, _ = b.decode_player(board, 1)
        # Note that the focus technically has no bearing on the win-state
        # so we don't attach it to the cache for MCTS to keep the number
        # of keys down.
        # JDD: UGH without this the invalid move selection goes up because keys conflict.

        # TODO: Add player index here? Store in board data after focus_index? This controls valid moves
        #       could there be a problem with cache conflicts? Are the policies the same for each player
        #       given the m/f/e?

        #  TRY ADDING PLAYER TO BOTH KEYS... We don't want the players sharing learnings during play/training.
        return "[ {}, {}, {} ]".format(m1, f1, e1)

    def getEndedStateKey(self, board):
        """conversion of board to a string format, required by MCTS for hashing."""
        b = MathBoard(self.width)
        # This is always called for the canonical board which means the
        # current player is always in player1 slot:
        e1, m1, _, _ = b.decode_player(board, 1)
        return "[ {}, {} ]".format(m1, e1)


def display(board):
    b = MathBoard(MathGame.width)
    e1, m1, _, _ = b.decode_player(board, 1)
    print("Player1 [move={}] [state={}]".format(m1, e1))
    e2, m2, _, _ = b.decode_player(board, -1)
    print("Player2 [move={}] [state={}]".format(m2, e2))

