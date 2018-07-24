from ..tree_node import LEFT
from ..expressions import AddExpression, BinaryExpression, ConstantExpression
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike
from .base_rule import BaseRule


class ConstantsSimplifyRule(BaseRule):
    """Given a binary operation on two constants, simplify to the resulting constant expression"""

    def getName(self):
        return "Simplify Constant Operation"

    def canApplyTo(self, node):
        self.needsRotation = None
        # Check simple case of left/right child binary op with constants
        if (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, ConstantExpression)
        ):
            return True

        # see if rotating the right node will result in a cosntant simplification
        # we can work with.
        if (
            isinstance(node.right, BinaryExpression)
            and node.right.getPriority() == node.getPriority()
        ):
            if node.right.left and isinstance(node.right.left, ConstantExpression):
                self.needsRotation = node.right
                return True

        # check for inverted case of node.right is constant, left is an add/subtract,
        # and left.right is another constant.  Rotating the left node will correct this
        # resulting in the desired left/right constants and a parent add.
        # This is returning bad results in some cases.  e.g. 21x^2 + 7y + 2y + 4 detects 2y + 4 as valid.
        if not isinstance(node, AddExpression):
            return False

        if not node.left or not isinstance(node.left, AddExpression):
            return False

        if not isinstance(node.right, ConstantExpression) or not isinstance(
            node.left.right, ConstantExpression
        ):
            return False

        self.needsRotation = node.left
        return True

    def applyTo(self, node):
        change = super().applyTo(node)
        if self.needsRotation is not None:
            self.needsRotation.rotate()

        change.saveParent()
        result = ConstantExpression(node.evaluate())
        return change.done(result)

