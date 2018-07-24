from ..tree_node import LEFT
from ..expressions import AddExpression, MultiplyExpression, ConstantExpression
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike
from .base_rule import BaseRule

# ### Associative Property
#
# Addition: `(a + b) + c = a + (b + c)`
#
#         (y) +            + (x)
#            / \          / \
#           /   \        /   \
#      (x) +     c  ->  a     + (y)
#         / \                / \
#        /   \              /   \
#       a     b            b     c
#
# Multiplication: `(ab)c = a(bc)`
#
#         (x) *            * (y)
#            / \          / \
#           /   \        /   \
#      (y) *     c  <-  a     * (x)
#         / \                / \
#        /   \              /   \
#       a     b            b     c
#
class AssociativeSwapRule(BaseRule):
    def getName(self):
        return "Associative Group"

    def canApplyTo(self, node):
        if isinstance(node.parent, AddExpression) and isinstance(node, AddExpression):
            return True

        if isinstance(node.parent, MultiplyExpression) and isinstance(
            node, MultiplyExpression
        ):
            return True

        return False

    def shouldApplyTo(self, node):
        if not super().shouldApplyTo(node):
            return False

        # If we're the right child, see if our grandparent has a like term on the right.
        if (
            node.parent
            and isAddSubtract(node.parent)
            and node.parent.getSide(node) == LEFT
        ):
            if termsAreLike(getTerm(node.right), getTerm(node.parent.right)):
                return True

        if isinstance(node.right, ConstantExpression) and isinstance(
            node.parent.right, ConstantExpression
        ):
            return True

        if not isAddSubtract(node.parent):
            return False

        lterm = getTerm(node.right)
        rterm = getTerm(node.parent.right)
        if not lterm or not rterm:
            return False

        if (lterm.variables[0] or rterm.variables[0]) and lterm.variables[
            0
        ] != rterm.variables[0]:
            return False

        if lterm.exponent and lterm.exponent != rterm.exponent:
            return False

        return True

    def applyTo(self, node):
        change = super().applyTo(node)
        node.rotate()
        change.done(node)
        return change
