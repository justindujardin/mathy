from ..expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
)
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike
from .base_rule import BaseRule

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
    def getName(self):
        return "Commutative Move"

    def canApplyTo(self, node):
        # Must be an add/multiply
        return isinstance(node, AddExpression) or isinstance(node, MultiplyExpression)

    def shouldApplyTo(self, node):
        if not super().shouldApplyTo(node):
            return False

        if not isinstance(node, MultiplyExpression):
            return False

        if not isinstance(node.right, ConstantExpression):
            return False

        # shouldn't apply to const*var (already in natural order.)
        if isinstance(node.right, VariableExpression) and isinstance(
            node.left, ConstantExpression
        ):
            return False

        # Chain of multiplications can be swapped.
        if isinstance(node.left, MultiplyExpression):
            return True

        if isinstance(node.left, VariableExpression):
            return True

        if isinstance(node.left, PowerExpression) and isinstance(
            node.left.left, VariableExpression
        ):
            return True

        return False

    def applyTo(self, node):
        change = super().applyTo(node)
        a = node.left
        b = node.right

        node.setRight(a)
        node.setLeft(b)

        change.done(node)
        return change

