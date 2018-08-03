import random
import math
import numpy
import time
from .core.expressions import (
    MathExpression,
    ConstantExpression,
    MultiplyExpression,
    PowerExpression,
    STOP,
    AddExpression,
    VariableExpression,
)
from .core.problems import ProblemGenerator
from .core.parser import ExpressionParser
from .core.util import termsAreLike, isAddSubtract, getTerms, getTerm, isSimpleTerm
from .core.rules import (
    BaseRule,
    AssociativeSwapRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    ConstantsSimplifyRule,
    CombineLikeTermsRule,
)
from .core.profiler import profile_start, profile_end
from .environment_state import EnvironmentState
from .math_actions import VisitBeforeAction, VisitAfterAction, MetaAction
import threading


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

    width = 128
    verbose = False
    draw = 0.0001
    max_moves = 15

    def __init__(self):
        self.parser = ExpressionParser()
        self.problems = ProblemGenerator()
        self.available_actions = [VisitAfterAction(), VisitBeforeAction()]
        self.available_rules = [
            CombineLikeTermsRule(),
            ConstantsSimplifyRule(),
            DistributiveFactorOutRule(),
            DistributiveMultiplyRule(),
            CommutativeSwapRule(),
            AssociativeSwapRule(),
        ]

    def getInitBoard(self, problem: str = None):
        """return a numpy encoded version of the input expression"""
        if problem is None:
            terms = random.randint(3, 3)
            problem = self.problems.simplify_multiple_terms(max_terms=terms)
        # TODO: Remove this stateful variable that is used mostly for printing out "{from} -> {to}" at game end
        # NOTE: If we store a plane for history per user we could do something like [first_state, last_n-2, last_n-1, last_n, current]
        self.expression_str = problem
        # problem = "14x + 7x"
        # print("\n\n\t\tNEXT: {}".format(problem))
        if len(list(problem)) > MathGame.width:
            raise ValueError(
                'Expression "{}" is too long for the current model to process. Max width is: {}'.format(
                    problem, MathGame.width
                )
            )
        board = EnvironmentState(MathGame.width).encode_board(problem)

        # NOTE: This is called for each episode, so it can be thought of like "onInitEpisode()"
        return board

    def write_draw(self, state):
        """Help spot errors in win conditons by always writing out draw values for review"""
        with open("draws.txt", "a") as file:
            file.write("{}\n".format(state))

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
        b = EnvironmentState(MathGame.width)
        features, move_count, focus_index, _, meta_counter, _ = b.decode_player(
            board, player
        )
        expression = self.parser.parse_features(features)
        token = self.getFocusToken(expression, focus_index)
        actions = self.available_actions + self.available_rules
        operation = actions[action]
        # searching = False

        # Enforce constraints to keep training time and complexity down?
        # - can't commutative swap immediately to return to previous state.
        # - can't focus on the same token twice without taking a valid other
        #   action inbetween
        # TODO: leaving these ideas here, but optimization made them less necessary
        if isinstance(operation, BaseRule) and operation.canApplyTo(token):
            change = operation.applyTo(token.rootClone())
            root = change.end.getRoot()
            out_features = self.parser.make_features(str(root))
            if not searching and MathGame.verbose and player == 1:
                print("[{}] {}".format(move_count, change.describe()))
            out_board = b.encode_player(
                board,
                player,
                out_features,
                move_count + 1,
                focus_index,
                meta_counter,
                action,
            )
        elif isinstance(operation, MetaAction):
            operation_result = operation.visit(self, expression, focus_index)
            if not searching and MathGame.verbose and player == 1:
                direction = (
                    "behind" if isinstance(operation, VisitBeforeAction) else "ahead"
                )
                print(
                    "[{}] 👀 Looking {} at: {}".format(
                        move_count,
                        direction,
                        self.getFocusToken(expression, operation_result),
                    )
                )
            out_board = b.encode_player(
                board,
                player,
                features,
                move_count + 1,
                operation_result,
                meta_counter + 1,
                action,
            )
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
            searching: True if called by MCTS simulations

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        b = EnvironmentState(MathGame.width)
        features, _, focus_index, _, meta_counter, _ = b.decode_player(board, player)
        expression = self.parser.parse_features(features)
        token = self.getFocusToken(expression, focus_index)
        actions = self.get_actions_for_expression(token)
        # NOTE: Below is verbose output showing which actions are valid.
        # if not searching:
        #     print_list = self.available_actions + self.available_rules
        #     [
        #         print(
        #             "Player{} action[{}][{}] = {}".format(
        #                 player, i, bool(a), type(print_list[i]) if a != 0 else ""
        #             )
        #         )
        #         for i, a in enumerate(actions)
        #     ]
        return actions

    def get_actions_for_expression(self, expression: MathExpression):
        actions = [0] * self.getActionSize()
        count = 0
        # Meta actions first
        for action in self.available_actions:
            if action.canVisit(self, expression):
                actions[count] = 1
            count = count + 1
        # Rules of numbers
        for rule in self.available_rules:
            if rule.canApplyTo(expression):
                actions[count] = 1
            count = count + 1

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
        b = EnvironmentState(MathGame.width)
        features, move_count, _, _, _, _ = b.decode_player(board, player)
        expression = self.parser.parse_features(features)
        # It's over if the expression is reduced to a single constant
        if (
            isinstance(expression, ConstantExpression)
            and expression.parent is None
            and expression.left is None
            and expression.right is None
        ):
            # print("Game done with constant win: ")
            # print("  start expression {}".format(self.expression_str))
            # print("  end expression {}".format(str(expression)))
            # eval = self.parser.parse(self.expression_str).evaluate()
            # found = expression.evaluate()
            # if eval != found and not math.isclose(eval, found):
            #     if not searching:
            #         print(
            #             "[LOSE] ERROR: reduced '{}' to constant, but evals differ. Expected '{}' but got '{}'!".format(
            #                 self.expression_str, eval, found
            #             )
            #         )
            #     return -1

            # Holy shit it won!
            if not searching and MathGame.verbose:
                print(
                    "\n[Player{}][WIN] {} => {}!".format(
                        player, self.expression_str, expression
                    )
                )
            else:
                pass
                # print(".")

            return 1

        # Check for simplification removes all like terms
        root = expression.getRoot()
        if isAddSubtract(root):
            term_nodes = getTerms(root)
            term_count = len(term_nodes)
            # Only handle up to two terms right now. Once it can
            # deal with "4x + 3x + 2" and "12x * 2x^3 * 4" we can
            # try expanding to arbitrary polynomial
            two_unlike = (
                term_count == 2 and termsAreLike(term_nodes[0], term_nodes[1]) == False
            )
            if term_count == 1 or two_unlike:
                t1 = getTerm(term_nodes[0])
                t2 = getTerm(term_nodes[1])
                one_coefficients = len(t1.coefficients)
                two_coefficients = len(t2.coefficients)
                # Make sure each term only has one coefficient, or else more simplification
                # can still be done.
                if one_coefficients < 2 and two_coefficients < 2:
                    # TODO: Compare constant/variable findings to verify their accuracy.
                    #       This helps until we're 100% confident in the parser/serializer consistency
                    #
                    # Holy shit it won!
                    if not searching and MathGame.verbose:
                        print(
                            "\n[Player{}][TERMWIN] {} => {}!".format(
                                player, self.expression_str, expression
                            )
                        )
                    return 1

        # A single term hanging out by itself (e.g. 24x, x^2)
        if isinstance(root, MultiplyExpression) or isinstance(root, PowerExpression):
            term_nodes = getTerms(root)
            if len(term_nodes) == 1 and isSimpleTerm(root):
                if not searching and MathGame.verbose:
                    print(
                        "\n[Player{}][TERMWIN] {} => {}!".format(
                            player, self.expression_str, expression
                        )
                    )
                return 1

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, it should be honored.
        if move_count > MathGame.max_moves:
            f2, move_count_other, _, _, _, _ = b.decode_player(board, player * -1)
            if move_count_other > MathGame.max_moves:
                if not searching:
                    e2 = self.parser.parse_features(f2)
                    draw_time = int(round(time.time() * 1000))
                    self.write_draw(
                        "[time={}][DRAW] ENDED WITH:\n\t input: {}\n\t 1: {}\n\t 2: {}\n".format(
                            draw_time, self.expression_str, expression, e2
                        )
                    )
                return MathGame.draw

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
        # print("gcf: {}".format(player))
        return EnvironmentState(MathGame.width).get_canonical_board(board, player)

    #
    def getCanonicalAgentState(self, board):
        # b = EnvironmentState(MathGame.width)
        # return b.slice_player_data(board, 1)
        return board

    def getAgentStateSize(self):
        """return shape (x,y) of agent state dimensions"""
        # 2 columns per player, the first for turn data, the second for text inputs
        return (4, MathGame.width)

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
        b = EnvironmentState(MathGame.width)
        # This is always called for the canonical board which means the
        # current player is always in player1 slot:
        features, m1, f1, _, _, _ = b.decode_player(board, 1)
        features_key = ",".join([str(f) for f in features])
        return "[{},{},{}]".format(m1, f1, features_key)

    def getEndedStateKey(self, board):
        """conversion of board to a string format, required by MCTS for hashing."""
        b = EnvironmentState(MathGame.width)
        # This is always called for the canonical board which means the
        # current player is always in player1 slot:
        features, m1, _, _, _, _ = b.decode_player(board, 1)
        features_key = ",".join([str(f) for f in features])
        return "[{},{}]".format(m1, features_key)


_parser = None


def display(board, player):
    global _parser
    if _parser is None:
        _parser = ExpressionParser()
    b = EnvironmentState(MathGame.width)
    features = b.decode_player(board, player)[0]
    expression = _parser.parse_features(features)
    expression_len = len(str(expression))
    width = 100
    if player == 1:
        buffer = " " * int(width / 2 - expression_len)
    elif player == -1:
        buffer = " " * int(width - expression_len)
    else:
        raise ValueError("invalid player index: {}".format(player))
    print("{}{}".format(buffer, expression))