from ..core.expressions import AddExpression, MultiplyExpression
from ..core.rule import BaseRule


class AssociativeSwapRule(BaseRule):
    r"""Associative Property
    Addition: `(a + b) + c = a + (b + c)`

             (y) +            + (x)
                / \          / \
               /   \        /   \
          (x) +     c  ->  a     + (y)
             / \                / \
            /   \              /   \
           a     b            b     c

     Multiplication: `(ab)c = a(bc)`

             (x) *            * (y)
                / \          / \
               /   \        /   \
          (y) *     c  <-  a     * (x)
             / \                / \
            /   \              /   \
           a     b            b     c
        """

    @property
    def name(self) -> str:
        return "Associative Group"

    @property
    def code(self) -> str:
        return "AG"

    def can_apply_to(self, node) -> bool:
        if isinstance(node.parent, AddExpression) and isinstance(node, AddExpression):
            return True

        if isinstance(node.parent, MultiplyExpression) and isinstance(
            node, MultiplyExpression
        ):
            return True

        return False

    def apply_to(self, node):
        change = super().apply_to(node)
        node.rotate()
        node.set_changed()
        change.done(node)
        return change
