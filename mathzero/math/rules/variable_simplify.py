from ..tree_node import LEFT
from ..expressions import AddExpression, VariableExpression, ConstantExpression, MultiplyExpression
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike
from .base_rule import BaseRule


class VariableSimplifyRule(BaseRule):
    """Combine like variables that are next to each other"""

    def getName(self):
        return "Simplify like variables"

    def canApplyTo(self, node):
        # Check simple case of left/right child binary op with single variables
        return (
            isAddSubtract(node)
            and isinstance(node.left, VariableExpression)
            and isinstance(node.right, VariableExpression)
            and node.left.identifier == node.right.identifier
        )
    def applyTo(self, node):
        change = super().applyTo(node)
        change.saveParent()
        result = MultiplyExpression(ConstantExpression(2), VariableExpression(node.left.identifier))
        return change.done(result)

