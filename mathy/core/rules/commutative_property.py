from ..expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
)
from ..util import (
    is_add_or_sub,
    is_const,
    terms_are_like,
    is_preferred_term_form,
)
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
    def __init__(self, preferred=True):
        # If false, terms that are in preferred order will not commute
        self.preferred = preferred

    @property
    def name(self):
        return "Commutative Swap"

    @property
    def code(self):
        return "CS"

    def can_apply_to(self, node):
        # Must be an add/multiply
        if isinstance(node, AddExpression):
            return True
        if not isinstance(node, MultiplyExpression):
            return False
        if self.preferred is False:
            # 4x won't commute to x * 4 if preferred is false because 4x is the preferred term order
            left_const = isinstance(node.left, ConstantExpression)
            right_var = isinstance(node.right, VariableExpression)
            if left_const and right_var:
                return False
        return True

    def apply_to(self, node):
        change = super().apply_to(node)
        a = node.left
        b = node.right

        node.set_right(a)
        node.set_left(b)

        change.done(node)
        return change