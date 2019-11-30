from typing import Optional, Tuple

from ..core.expressions import (
    AddExpression,
    BinaryExpression,
    ConstantExpression,
    MultiplyExpression,
    VariableExpression,
)
from ..core.rule import BaseRule


class ConstantsSimplifyRule(BaseRule):
    """Given a binary operation on two constants, simplify to the resulting
    constant expression"""

    @property
    def name(self):
        return "Constant Arithmetic"

    @property
    def code(self):
        return "CA"

    POS_SIMPLE: str = "simple"
    POS_SIMPLE_VAR_MULT: str = "simple_var_multiply"
    POS_CHAINED_RIGHT: str = "chained_right"
    POS_CHAINED_RIGHT_LEFT: str = "chained_right_left"
    POS_CHAINED_RIGHT_LEFT_LEFT: str = "chained_right_left_left"
    POS_CHAINED_LEFT_LEFT_RIGHT: str = "chained_left_left_right"
    POS_CHAINED_RIGHT_DEEP: str = "chained_right_deep"

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

        # Check for const * var * const
        # (4n * 2) + 3
        if (
            isinstance(node, MultiplyExpression)
            and isinstance(node.left, MultiplyExpression)
            and isinstance(node.left.left, ConstantExpression)
            and isinstance(node.left.right, VariableExpression)
            and isinstance(node.right, ConstantExpression)
        ):
            return ConstantsSimplifyRule.POS_SIMPLE_VAR_MULT, node.left.left, node.right

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
        # "(7 * 10y^3) * x"
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

        # Check for a continuation to the right
        # "(7q * 10y^3) * x"
        if (
            isinstance(node, MultiplyExpression)
            and isinstance(node.left, MultiplyExpression)
            and isinstance(node.left.left, ConstantExpression)
            and isinstance(node.right, MultiplyExpression)
            and isinstance(node.right.left, ConstantExpression)
        ):
            return (
                ConstantsSimplifyRule.POS_CHAINED_RIGHT_LEFT,
                node.left.left,
                node.right.left,
            )

        # Check for variable terms with constants on the left and right
        # "792z^4 * 490f * q^3"
        #   ^--------^
        if (
            isinstance(node, MultiplyExpression)
            and isinstance(node.left, MultiplyExpression)
            and isinstance(node.left.left, ConstantExpression)
            and isinstance(node.right, MultiplyExpression)
            and isinstance(node.right.left, MultiplyExpression)
            and isinstance(node.right.left.left, ConstantExpression)
        ):
            return (
                ConstantsSimplifyRule.POS_CHAINED_RIGHT_LEFT_LEFT,
                node.left.left,
                node.right.left.left,
            )

        # Check for variable terms with constants nested on the left and right
        # "(u^3 * 36c^6) * 7u^3"
        #         ^--------^
        if (
            isinstance(node, MultiplyExpression)
            and isinstance(node.left, MultiplyExpression)
            and isinstance(node.left.right, MultiplyExpression)
            and isinstance(node.left.right.left, ConstantExpression)
            and isinstance(node.right, MultiplyExpression)
            and isinstance(node.right.left, ConstantExpression)
        ):
            return (
                ConstantsSimplifyRule.POS_CHAINED_LEFT_LEFT_RIGHT,
                node.left.right.left,
                node.right.left,
            )

        return None

    def can_apply_to(self, node):
        return self.get_type(node) is not None

    def apply_to(self, node):
        change = super().apply_to(node)
        arrangement, left_const, right_const = self.get_type(node)
        change.save_parent()
        if arrangement == ConstantsSimplifyRule.POS_SIMPLE:
            result = ConstantExpression(node.evaluate())
        if arrangement == ConstantsSimplifyRule.POS_SIMPLE_VAR_MULT:
            assert isinstance(node, MultiplyExpression)
            assert isinstance(node.left.right, VariableExpression)
            value = ConstantExpression(
                MultiplyExpression(left_const, right_const).evaluate()
            )
            result = MultiplyExpression(value, node.left.right)
        elif arrangement == ConstantsSimplifyRule.POS_CHAINED_LEFT_LEFT_RIGHT:
            assert isinstance(node, MultiplyExpression)
            value = ConstantExpression(
                MultiplyExpression(left_const, right_const).evaluate()
            )
            result = MultiplyExpression(
                node.left.left,
                MultiplyExpression(
                    MultiplyExpression(value, node.left.right.right), node.right.right
                ),
            )

        elif arrangement == ConstantsSimplifyRule.POS_CHAINED_RIGHT_LEFT:
            value = ConstantExpression(
                MultiplyExpression(left_const, right_const).evaluate()
            )
            value = MultiplyExpression(value, node.left.right)
            result = MultiplyExpression(value, node.right.right)
        elif arrangement == ConstantsSimplifyRule.POS_CHAINED_RIGHT_LEFT_LEFT:
            value = ConstantExpression(
                MultiplyExpression(left_const, right_const).evaluate()
            )
            value = MultiplyExpression(value, node.left.right)
            result = MultiplyExpression(
                value, MultiplyExpression(node.right.left.right, node.right.right)
            )
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
