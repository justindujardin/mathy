from ..tree import LEFT
from ..expressions import AddExpression, MultiplyExpression, ConstantExpression
from ..util import is_add_or_sub, is_const, terms_are_like
from ..rule import BaseRule

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

    @property
    def code(self):
        return "AG"

    def can_apply_to(self, node):
        if isinstance(node.parent, AddExpression) and isinstance(node, AddExpression):
            return True

        if isinstance(node.parent, MultiplyExpression) and isinstance(node, MultiplyExpression):
            return True

        return False

    def apply_to(self, node):
        change = super().apply_to(node)
        node.rotate()
        change.done(node)
        return change
