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
    @property
    def name(self):
        return "Associative Group"

    def canApplyTo(self, node):
        if isinstance(node.parent, AddExpression) and isinstance(node, AddExpression):
            return True

        if isinstance(node.parent, MultiplyExpression) and isinstance(node, MultiplyExpression):
            return True

        return False

    def applyTo(self, node):
        change = super().applyTo(node)
        node.rotate()
        change.done(node)
        return change
