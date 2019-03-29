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
from collections import namedtuple


class MetaAction:
    pass


FocusActionChoice = namedtuple("FocusActionChoice", ["node", "index"])


class VisitBeforeAction(MetaAction):
    @property
    def code(self):
        return "BK"

    def visit(self, game, expression: MathExpression, focus_index: int):
        """Visit all the nodes in the expression, and move the focus to the
        nearest one that has a lower focus index, wrapping around to higher numbers
        only if no lower ones are found."""
        nodes = expression.getRoot().toList("preorder")
        all_choices = []
        for i, node in enumerate(nodes):
            moves = game.get_actions_for_node(node)
            # zero out the first two Meta actions so they
            # don't get counted in the argmax below
            moves[0] = moves[1] = 0
            non_meta = numpy.argmax(moves)
            # If we found a non-meta action that can be applied, stop here
            if not node or non_meta > 0:
                all_choices.append(FocusActionChoice(node, i))
        lower = [l.index for l in all_choices if l.index < focus_index]
        higher = [l.index for l in all_choices if l.index > focus_index]
        if len(lower) == 0 and len(higher) == 0:
            focus_index = focus_index - 1
            if focus_index < 0:
                focus_index = len(nodes) - 1
            return focus_index
        return lower[0] if len(lower) > 0 else higher[-1]


class VisitAfterAction(MetaAction):
    @property
    def code(self):
        return "FW"

    def visit(self, game, expression: MathExpression, focus_index: int):
        """Visit all the nodes in the expression, and move the focus to the
        nearest one that has a higher focus index, wrapping around to lower numbers
        only if no higher ones are found."""
        nodes = expression.getRoot().toList("preorder")
        all_choices = []
        for i, node in enumerate(nodes):
            moves = game.get_actions_for_node(node)
            # skip over the first two moves when argmax'ing because they're meta actions
            non_meta = numpy.argmax(moves[2:])
            # If we found a non-meta action that can be applied, stop here
            if not node or non_meta > 0:
                all_choices.append(FocusActionChoice(node, i))
        lower = [l.index for l in all_choices if l.index < focus_index]
        higher = [l.index for l in all_choices if l.index > focus_index]
        if len(lower) == 0 and len(higher) == 0:
            focus_index = focus_index + 1
            if focus_index >= len(nodes):
                focus_index = 0
            return focus_index

        return higher[0] if len(higher) > 0 else lower[-1]
