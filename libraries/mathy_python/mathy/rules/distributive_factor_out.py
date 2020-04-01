from typing import List, Optional, Tuple

from ..core.expressions import (
    AddExpression,
    MathExpression,
    MultiplyExpression,
    SubtractExpression,
)
from ..core.rule import BaseRule
from ..util import (
    TermEx,
    factor_add_terms_ex,
    get_term_ex,
    is_add_or_sub,
    make_term,
    terms_are_like,
    unlink,
)


class DistributiveFactorOutRule(BaseRule):
    r"""Distributive Property
        `ab + ac = a(b + c)`

         The distributive property can be used to expand out expressions
         to allow for simplification, as well as to factor out common properties
         of terms.

         **Factor out a common term**

         This handles the `ab + ac` conversion of the distributive property, which
         factors out a common term from the given two addition operands.

                   +               *
                  / \             / \
                 /   \           /   \
                /     \    ->   /     \
               *       *       a       +
              / \     / \             / \
             a   b   a   c           b   c
    """
    constants: bool

    def __init__(self, constants=False):
        # If true, will factor common numbers out of a const+const expression
        self.constants = constants

    POS_SIMPLE = "simple"
    POS_CHAINED_BOTH = "chained_both"
    POS_CHAINED_LEFT = "chained_left"
    POS_CHAINED_LEFT_RIGHT = "chained_left_right"
    POS_CHAINED_RIGHT_LEFT = "chained_right_left"
    POS_CHAINED_RIGHT = "chained_right"

    @property
    def name(self):
        return "Distributive Factoring"

    @property
    def code(self):
        return "DF"

    def get_type(self, node) -> Optional[Tuple[str, TermEx, TermEx]]:
        """Determine the configuration of the tree for this transformation.

        Support the three types of tree configurations:
         - Simple is where the node's left and right children are exactly
           terms linked by an add operation.
         - Chained Left is where the node's left child is a term, but the right
           child is another add operation. In this case the left child
           of the next add node is the target.
         - Chained Right is where the node's right child is a term, but the left
           child is another add operation. In this case the right child
           of the child add node is the target.

        Structure:
         - Simple
            * node(add),node.left(term),node.right(term)
         - Chained Left
            * node(add),node.left(term),node.right(add),node.right.left(term)
         - Chained Right
            * node(add),node.right(term),node.left(add),node.left.right(term)
        """
        if not isinstance(node, AddExpression):
            return None
        # Left node in both cases is always resolved as a term.
        left_term = get_term_ex(node.left)
        right_term = get_term_ex(node.right)

        # No terms found for either child
        if left_term is None and right_term is None:
            # Check for the rare case where both terms are chained.
            # This happens when forcing the associative groups into
            # a certain form. It's not usually useful, but it's a
            # valid thing to do.

            if isinstance(node.right, AddExpression):
                right_term = get_term_ex(node.right.left)
            if right_term is None or right_term.variable is None:
                return None

            if isinstance(node.left, AddExpression):
                left_term = get_term_ex(node.left.right)
            if left_term is None or left_term.variable is None:
                return None
            return DistributiveFactorOutRule.POS_CHAINED_BOTH, left_term, right_term

        # Simplest case of each child being a term exactly.
        if left_term is not None and right_term is not None:
            return DistributiveFactorOutRule.POS_SIMPLE, left_term, right_term

        # Left child is a term
        if left_term is not None:
            # TODO: I'm not sure why I had this restriction here.
            # TODO: add a comment about it when you remember.
            if left_term.variable is None:
                return None
            if isinstance(node.right, AddExpression):
                right_term = get_term_ex(node.right.left)
            if right_term is not None:
                if right_term.variable is None:
                    return None
                return (
                    DistributiveFactorOutRule.POS_CHAINED_RIGHT,
                    left_term,
                    right_term,
                )

            # check inside another group
            if isinstance(node.right, AddExpression) and isinstance(
                node.right.left, AddExpression
            ):
                right_term = get_term_ex(node.right.left.left)
            if right_term is None or right_term.variable is None:
                return None
            return (
                DistributiveFactorOutRule.POS_CHAINED_RIGHT_LEFT,
                left_term,
                right_term,
            )

        # Right child is a term
        if right_term is not None:
            # TODO: I'm not sure why I had this restriction here.
            # TODO: add a comment about it when you remember.
            if right_term.variable is None:
                return None
            if isinstance(node.left, AddExpression):
                left_term = get_term_ex(node.left.right)
            if left_term is not None:
                if left_term.variable is None:
                    return None
                return DistributiveFactorOutRule.POS_CHAINED_LEFT, left_term, right_term

            # check inside another group
            if isinstance(node.left, AddExpression) and isinstance(
                node.left.right, AddExpression
            ):
                left_term = get_term_ex(node.left.right.right)
            if left_term is None or left_term.variable is None:
                return None
            return (
                DistributiveFactorOutRule.POS_CHAINED_LEFT_RIGHT,
                left_term,
                right_term,
            )

        return None

    def can_apply_to(self, node):
        type_tuple = self.get_type(node)
        if type_tuple is None:
            return False
        type, l_term, r_term = type_tuple
        # Don't try factoring out terms with no variables, e.g "4 + 84"
        if (
            self.constants is False
            and l_term.variable is None
            and r_term.variable is None
        ):
            return False

        f = factor_add_terms_ex(l_term, r_term)
        if not f:
            return False

        if f.best == 1 and not f.variable and not f.exponent:
            return False

        return True

    def apply_to(self, node):
        change = super().apply_to(node).save_parent()
        tree_position, left_term, right_term = self.get_type(node)
        assert left_term is not None
        assert right_term is not None
        factors = factor_add_terms_ex(left_term, right_term)
        a = make_term(factors.best, factors.variable, factors.exponent)
        b = make_term(factors.left, factors.leftVariable, factors.leftExponent)
        c = make_term(factors.right, factors.rightVariable, factors.rightExponent)
        inside = (
            AddExpression(b, c)
            if isinstance(node, AddExpression)
            else SubtractExpression(b, c)
        )
        # NOTE: we swap the output order of the extracted
        #       common factor and what remains to prefer
        #       ordering that can be expressed without an
        #       explicit multiplication symbol.
        result = MultiplyExpression(inside, a)
        result.all_changed()

        # Fix the links to existing nodes on the left side of the result
        left_positions = [
            DistributiveFactorOutRule.POS_CHAINED_LEFT,
            DistributiveFactorOutRule.POS_CHAINED_BOTH,
        ]
        if tree_position in left_positions:
            # Because in the chained mode we extract node.left.right, the other
            # child is the remainder we want to be sure to preserve.
            # e.g. "(4 + p) + p" we need to keep "4"
            keep_child = node.left.left
            result = AddExpression(keep_child, result)

        # Fix the links to existing nodes on the left-right side of the result
        if tree_position == DistributiveFactorOutRule.POS_CHAINED_LEFT_RIGHT:
            keep_child = AddExpression(node.left.left, node.left.right.left)
            result = AddExpression(keep_child, result)

        # Fix the links to existing nodes on the right-left side of the result
        if tree_position == DistributiveFactorOutRule.POS_CHAINED_RIGHT_LEFT:
            keep_child = AddExpression(node.right.left.right, node.right.right)
            result = AddExpression(result, keep_child)

        # Fix the links to existing nodes on the right side of the result
        right_positions = [
            DistributiveFactorOutRule.POS_CHAINED_RIGHT,
            DistributiveFactorOutRule.POS_CHAINED_BOTH,
        ]
        if tree_position in right_positions:
            # Because in the chained mode we extract node.right.left, the other
            # child is the remainder we want to be sure to preserve.
            # e.g. "p + (p + 2x)" we need to keep 2x
            keep_child = node.right.right
            result = AddExpression(result, keep_child)
        change.done(result)
        return change
