import random
import math
import numpy
from .core.expressions import (
    MathExpression,
    ConstantExpression,
    STOP,
    AddExpression,
    VariableExpression,
)


class MetaAction:
    def getNthTokenFromLeft(
        self, expression: MathExpression, from_start: int
    ) -> MathExpression:
        """Get the token that is n tokens away from the start of the expression"""
        count = 0
        result = None

        def visit_fn(node, depth, data):
            nonlocal result, count
            result = node
            if count == from_start:
                return STOP
            count = count + 1

        expression.visitInorder(visit_fn)
        return result

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
    def canVisit(self, game, expression: MathExpression) -> bool:
        count, found = self.count_nodes(expression)
        return bool(found > 1 and count > 1)

    def visit(self, game, expression: MathExpression, focus_index: int):
        new_index = focus_index - 1
        new_token = self.getNthTokenFromLeft(expression, new_index)
        while True:
            # Ignore the first two meta moves.
            moves = game.get_actions_for_expression(new_token)[2:]
            non_meta = numpy.argmax(moves)
            if not new_token or non_meta > 0 or new_index == 0:
                break
            new_index -= 1
            new_token = self.getNthTokenFromLeft(new_token, new_index)
        return new_index


class VisitAfterAction(MetaAction):
    def canVisit(self, game, expression: MathExpression) -> bool:
        count, found = self.count_nodes(expression)
        return bool(found >= 0 and found < count - 2)

    def visit(self, game, expression: MathExpression, focus_index: int):
        new_index = focus_index + 1
        new_token = self.getNthTokenFromLeft(expression, new_index)
        count, _ = self.count_nodes(expression)
        while True:
            # Ignore the first two meta moves.
            moves = game.get_actions_for_expression(new_token)[2:]
            non_meta = numpy.argmax(moves)
            if not new_token or non_meta > 0 or new_index == (count - 2):
                break
            new_index += 1
            new_token = self.getNthTokenFromLeft(new_token, new_index)
        return new_index

