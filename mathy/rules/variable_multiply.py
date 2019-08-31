from typing import Optional, Tuple

from ..core.expressions import (
    AddExpression,
    ConstantExpression,
    MathExpression,
    MultiplyExpression,
    PowerExpression,
    VariableExpression,
)
from .rule import BaseRule
from .helpers import TermEx, get_term_ex


class VariableMultiplyRule(BaseRule):
    r"""
    This restates `x^b * x^d` as `x^(b + d)` which has the effect of isolating
    the exponents attached to the variables, so they can be combined.

        1. When there are two terms with the same base being multiplied together, their
           exponents are added together. "x * x^3" = "x^4" because "x = x^1" so
           "x^1 * x^3 = x^(1 + 3) = x^4"

        TODO: 2. When there is a power raised to another power, they can be combined by
                 multiplying the exponents together. "x^(2^2) = x^4"

    The rule identifies terms with explicit and implicit powers, so the following
    transformations are all valid:

    Explicit powers: x^b * x^d = x^(b+d)

            *
           / \
          /   \          ^
         /     \    =   / \
        ^       ^      x   +
       / \     / \        / \
      x   b   x   d      b   d


    Implicit powers: x * x^d = x^(1 + d)

            *
           / \
          /   \          ^
         /     \    =   / \
        x       ^      x   +
               / \        / \
              x   d      1   d

    """
    POS_SIMPLE = "simple"
    POS_CHAINED = "chained"

    @property
    def name(self):
        return "Variable Multiplication"

    @property
    def code(self):
        return "VM"

    def get_type(self, node: MathExpression) -> Optional[Tuple[str, TermEx, TermEx]]:
        """Determine the configuration of the tree for this transformation.

        Support two types of tree configurations:
         - Simple is where the node's left and right children are exactly
           terms that can be multiplied together.
         - Chained is where the node's left child is a term, but the right
           child is a continuation of a more complex term, as indicated by
           the presence of another Multiply node. In this case the left child
           of the next multiply node is the target.

        Structure:
         - Simple node(mult),node.left(term),node.right(term)
         - Chained node(mult),node.left(term),node.right(mult),node.right.left(term)
        """
        if not isinstance(node, MultiplyExpression):
            return None
        # Left node in both cases is always resolved as a term.
        left_term = get_term_ex(node.left)
        if left_term is None or left_term.variable is None:
            return None
        chained = False
        right_term = get_term_ex(node.right)
        if right_term is None and isinstance(node.right, MultiplyExpression):
            chained = True
            right_term = get_term_ex(node.right.left)
        if right_term is None or right_term.variable is None:
            return None
        # Variables need to match because we're summing exponents.
        #
        # e.g. "4x * 2y * 5x * 3y = 120 * x^2 * y^2"
        if left_term.variable != right_term.variable:
            return None
        if chained is True:
            return VariableMultiplyRule.POS_CHAINED, left_term, right_term
        return VariableMultiplyRule.POS_SIMPLE, left_term, right_term

    def can_apply_to(self, node: MathExpression):
        if not isinstance(node, MultiplyExpression):
            return False
        type_tuple = self.get_type(node)
        if type_tuple is None:
            return False
        return True

    def apply_to(self, node: MathExpression):
        change = super().apply_to(node).save_parent()
        type_tuple = self.get_type(node)
        assert type_tuple is not None
        tree_position, left_term, right_term = type_tuple
        assert left_term is not None
        assert right_term is not None

        left_exp = 1 if left_term.exponent is None else left_term.exponent
        right_exp = 1 if right_term.exponent is None else right_term.exponent

        # exponential inside term, e.g. given input "2x^2 * 4x" is "(2 + 1)"
        exponent_sum = AddExpression(
            ConstantExpression(left_exp), ConstantExpression(right_exp)
        )
        # The whole power term, e.g. given input "2x^2 * 4x" is "x^(2 + 1)"
        power_term = PowerExpression(
            VariableExpression(left_term.variable), exponent_sum
        )
        result: MathExpression = power_term
        # If either term has a coefficient we have to include a second term
        # in the output
        has_left_coefficient = left_term.coefficient is not None
        has_right_coefficient = right_term.coefficient is not None
        if has_left_coefficient or has_right_coefficient:
            left_const = 1 if left_term.coefficient is None else left_term.coefficient
            right_const = (
                1 if right_term.coefficient is None else right_term.coefficient
            )
            coefficient_term: MathExpression
            if has_left_coefficient and has_right_coefficient:
                # "(2 * 4)" both terms have coefficients
                coefficient_term = MultiplyExpression(
                    ConstantExpression(left_const), ConstantExpression(right_const)
                )
            else:
                # "2" only one term had a coefficient
                const_value = left_const if has_left_coefficient else right_const
                coefficient_term = ConstantExpression(const_value)
            # "(2 * 4) * x^(2 + 1)"
            result = MultiplyExpression(coefficient_term, power_term)

        # Mark all nodes in the result as having changed
        result.all_changed()

        # chained type has to fixup the tree to keep the chain unbroken
        if tree_position == VariableMultiplyRule.POS_CHAINED:
            # Because in the chained mode we extract node.right.left, the other
            # child is the remainder we want to be sure to preserve.
            # e.g. "p * p * 2x" we need to keep 2x
            keep_child = node.right.right
            result = MultiplyExpression(result, keep_child)
        change.done(result)
        return change
