from ..tree import LEFT
from ..expressions import AddExpression, BinaryExpression, ConstantExpression
from ..util import is_add_or_sub, is_const, terms_are_like
from ..rule import BaseRule


class ConstantsSimplifyRule(BaseRule):
    """Given a binary operation on two constants, simplify to the resulting constant expression"""

    @property
    def name(self):
        return "Constant Arithmetic"

    @property
    def code(self):
        return "CA"

    def can_apply_to(self, node):
        # Check simple case of left/right child binary op with constants
        return (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, ConstantExpression)
        )

    def apply_to(self, node):
        change = super().apply_to(node)
        change.save_parent()
        result = ConstantExpression(node.evaluate())
        result.set_changed()
        return change.done(result)

