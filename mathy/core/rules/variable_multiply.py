from ..expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
    MathExpression,
)
from ..util import is_add_or_sub, get_term
from ..rule import BaseRule


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

    def get_type(self, node):
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
        left_var, _ = self.get_term_components(node.left)
        if left_var is None:
            return None

        chained = False
        right_interest = node.right
        if isinstance(right_interest, MultiplyExpression):
            chained = True
            right_interest = right_interest.left
        right_var, _ = self.get_term_components(right_interest)
        if right_var is None:
            return None
        if chained is True:
            return VariableMultiplyRule.POS_CHAINED
        return VariableMultiplyRule.POS_SIMPLE

    def get_term_components(self, node: MathExpression):
        """Return a tuple of (variable, exponent) for the given node, or
        (None, None) if the node is invalid for this rule."""
        # A power expression with variable/constant children is OKAY
        if isinstance(node, PowerExpression):
            if isinstance(node.left, VariableExpression) and isinstance(
                node.right, ConstantExpression
            ):
                return (node.left.identifier, node.right.value)
        # Single variable is OKAY "x"
        if isinstance(node, VariableExpression):
            return (node.identifier, 1)
        # # Is the node a simple multiply with a left that is a variable? "x"
        # if node is not None and isinstance(node.left, VariableExpression):
        #     return (node.left.identifier, 1)
        return (None, None)

    def can_apply_to(self, node):

        if not isinstance(node, MultiplyExpression):
            return False
        tree_position = self.get_type(node)
        if tree_position is None:
            return False
        return True

    def apply_to(self, node):
        # tree_position = self.get_type(node)
        # if tree_position is None:
        #     raise ValueError("invalid node for rule, call canApply first.")
        change = super().apply_to(node).save_parent()
        # Left node in both cases is always resolved as a term.
        left_variable, left_exp = self.get_term_components(node.left)
        if left_variable is None:
            raise ValueError("invalid rule application")

        chained = False
        right_interest = node.right
        if isinstance(right_interest, MultiplyExpression):
            chained = True
            right_interest = right_interest.left
        right_variable, right_exp = self.get_term_components(right_interest)
        if right_variable is None:
            raise ValueError("invalid rule application")

        # If the variables don't match
        variable = left_variable
        if left_variable != right_variable:
            variable = left_variable + right_variable
        inside = AddExpression(
            ConstantExpression(left_exp), ConstantExpression(right_exp)
        )
        # If both powers are 1 and the variables don't match, drop from the output
        if left_exp == 1 and right_exp == 1 and left_variable != right_variable:
            result = VariableExpression(variable)
        else:
            result = PowerExpression(VariableExpression(variable), inside)

        # chained type has to fixup the tree to keep the chain unbroken
        if chained is True:
            # Because in the chained mode we extract node.right.left, the other
            # child is the remainder we want to be sure to preserve.
            # e.g. "p * p * 2x" we need to keep 2x
            keep_child = node.right.right
            result = MultiplyExpression(result, keep_child)
        change.done(result)
        return change
