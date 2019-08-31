from typing import Optional, Tuple

from ..core.expressions import (
    AddExpression,
    BinaryExpression,
    ConstantExpression,
    MultiplyExpression,
)
from .rule import BaseRule


class ConstantsSimplifyRule(BaseRule):
    """Given a binary operation on two constants, simplify to the resulting
    constant expression"""

    @property
    def name(self):
        return "Constant Arithmetic"

    @property
    def code(self):
        return "CA"

    POS_SIMPLE = "simple"
    POS_CHAINED_RIGHT = "chained_right"
    POS_CHAINED_RIGHT_DEEP = "chained_right_deep"

    def get_type(
        self, node
    ) -> Optional[Tuple[str, ConstantExpression, ConstantExpression]]:
        """Determine the configuration of the tree for this transformation.

        Support the three types of tree configurations:
         - Simple is where the node's left and right children are exactly
           constants linked by an add operation.
         - Chained Right is where the node's left child is a constant, but the right
           child is another binary operation of the same type. In this case the left
           child of the next binary node is the target.

        Structure:
         - Simple
            * node(add),node.left(const),node.right(const)
         - Chained Right
            * node(add),node.left(const),node.right(add),node.right.left(const)
         - Chained Right Deep
            * node(add),node.left(const),node.right(add),node.right.left(const)
        """
        # Check simple case of left/right child binary op with constants
        # (4 * 2) + 3
        if (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, ConstantExpression)
        ):
            return ConstantsSimplifyRule.POS_SIMPLE, node.left, node.right

        # Check for a continuation to the right that's more than one node
        # e.g. "5 * (8h * t)" = "40h * t"
        if (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, BinaryExpression)
            and isinstance(node.right.left, BinaryExpression)
            and isinstance(node.right.left.left, ConstantExpression)
        ):
            # Add/Multiply continuations are okay
            if (
                isinstance(node, AddExpression)
                and isinstance(node.right, AddExpression)
                and isinstance(node.right.left, AddExpression)
                or isinstance(node, MultiplyExpression)
                and isinstance(node.right, MultiplyExpression)
                and isinstance(node.right.left, MultiplyExpression)
            ):
                return (
                    ConstantsSimplifyRule.POS_CHAINED_RIGHT_DEEP,
                    node.left,
                    node.right.left.left,
                )

        # Check for a continuation to the right
        if (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, BinaryExpression)
            and isinstance(node.right.left, ConstantExpression)
        ):
            # Add/Multiply continuations are okay
            if (
                isinstance(node, AddExpression)
                and isinstance(node.right, AddExpression)
                or isinstance(node, MultiplyExpression)
                and isinstance(node.right, MultiplyExpression)
            ):
                return (
                    ConstantsSimplifyRule.POS_CHAINED_RIGHT,
                    node.left,
                    node.right.left,
                )

        return None

    def get_operands(self, node):
        # Check simple case of left/right child binary op with constants

        # TODO: Need to support these forms (binomial, complex) envs are failing
        #       because of unmatched rules.

        # Examples where two constants were siblings, but were not offered for
        # simplification.
        #
        # -- cs -- -- ag -- | 12 | 003 | commutative swap          | 5 * (8h * t)
        # -- cs -- -- ag -- | 10 | 003 | commutative swap          | 5 * (8t * h)

        return (
            isinstance(node, BinaryExpression)
            and isinstance(node.left, ConstantExpression)
            and isinstance(node.right, ConstantExpression)
        )

    def can_apply_to(self, node):
        return self.get_type(node) is not None

    def apply_to(self, node):
        change = super().apply_to(node)
        arrangement, left_const, right_const = self.get_type(node)
        change.save_parent()
        if arrangement == ConstantsSimplifyRule.POS_SIMPLE:
            result = ConstantExpression(node.evaluate())
        elif arrangement == ConstantsSimplifyRule.POS_CHAINED_RIGHT:
            if isinstance(node, AddExpression):
                value = ConstantExpression(
                    AddExpression(left_const, right_const).evaluate()
                )
                result = AddExpression(value, node.right.right)
            elif isinstance(node, MultiplyExpression):
                value = ConstantExpression(
                    MultiplyExpression(left_const, right_const).evaluate()
                )
                result = MultiplyExpression(value, node.right.right)
            else:
                raise NotImplementedError(
                    f"can't deal with operand of {type(node)} type"
                )
        elif arrangement == ConstantsSimplifyRule.POS_CHAINED_RIGHT_DEEP:
            if isinstance(node, AddExpression):
                value = ConstantExpression(
                    AddExpression(left_const, right_const).evaluate()
                )
                result = AddExpression(
                    AddExpression(value, node.right.left.right), node.right.right
                )
            elif isinstance(node, MultiplyExpression):
                value = ConstantExpression(
                    MultiplyExpression(left_const, right_const).evaluate()
                )
                result = MultiplyExpression(
                    MultiplyExpression(value, node.right.left.right), node.right.right
                )
            else:
                raise NotImplementedError(
                    f"can't deal with operand of {type(node)} type"
                )
        result.set_changed()
        return change.done(result)
