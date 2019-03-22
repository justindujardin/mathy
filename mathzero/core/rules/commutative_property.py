from ..expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
)
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike
from ..rule import BaseRule

# ### Commutative Property
#
#
# For Addition: `a + b = b + a`
#
#             +                  +
#            / \                / \
#           /   \     ->       /   \
#          /     \            /     \
#         a       b          b       a
#
# For Multiplication: `ab = ba`
#
#             *                  *
#            / \                / \
#           /   \     ->       /   \
#          /     \            /     \
#         a       b          b       a
#
class CommutativeSwapRule(BaseRule):
    @property
    def name(self):
        return "Commutative Swap"

    @property
    def code(self):
        return "CS"

    def canApplyTo(self, node):
        # Must be an add/multiply
        return isinstance(node, AddExpression) or isinstance(node, MultiplyExpression)

    def applyTo(self, node):
        change = super().applyTo(node)
        a = node.left
        b = node.right

        node.setRight(a)
        node.setLeft(b)

        change.done(node)
        return change
