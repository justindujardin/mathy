from ..tree_node import LEFT
from ..expressions import (
    AddExpression,
    VariableExpression,
    ConstantExpression,
    MultiplyExpression,
)
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike, makeTerm
from .base_rule import BaseRule
import numpy


class CombineLikeTermsRule(BaseRule):
    """Combine like variables that are next to each other"""

    def getName(self):
        return "Simplify like variables"

    def canApplyTo(self, node):
        # Check simple case of left/right child binary op with single variables
        is_add_sub = isAddSubtract(node)
        if not is_add_sub and not isinstance(node, MultiplyExpression):
            return False

        # TODO: I think this restriction could be lifted, but this keeps the code simple.
        # Each child must resolve to a term with one variable and coefficient
        l_term = getTerm(node.left)
        if (
            l_term == False
            or len(l_term.variables) > 1
            or len(l_term.coefficients) != 1
        ):
            return False
        r_term = getTerm(node.right)
        if (
            r_term == False
            or len(r_term.variables) > 1
            or len(r_term.coefficients) != 1
        ):
            return False

        # If there are two variables and they don't match, no go.
        l_vars = l_term.variables
        r_vars = r_term.variables
        if (len(l_vars) > 0 and len(r_vars) > 0) and l_vars[0] != r_vars[0]:
            return False

        # TODO: Verify this.
        # Exponents must match with adding two terms? 4x^2 + 5x^2
        l_exp = l_term.exponent
        r_exp = r_term.exponent
        if is_add_sub and l_exp is not None and r_exp is not None and l_exp != r_exp:
            return False
        return True

    def applyTo(self, node):
        change = super().applyTo(node)
        change.saveParent()
        l_term = getTerm(node.left)
        r_term = getTerm(node.right)
        coefficients = l_term.coefficients + r_term.coefficients
        l_vars = l_term.variables
        r_vars = r_term.variables
        if isAddSubtract(node):
            coefficient = numpy.sum(coefficients)
            exponent = l_term.exponent
        else:
            coefficient = numpy.product(coefficients)

            # If there is a variable, the implicit exponent to be summed is 1, otherwise 0
            implicit_l_exp = len(l_vars)
            implicit_r_exp = len(r_vars)
            
            left_exp = l_term.exponent if l_term.exponent else implicit_l_exp
            right_exp = r_term.exponent if r_term.exponent else implicit_r_exp
            exponent = left_exp + right_exp

        variable = (
            l_vars[0] if len(l_vars) > 0 else r_vars[0] if len(r_vars) > 0 else None
        )
        # Implicitly everything has an exponent of ^1, so leave it off.
        if exponent is not None and exponent == 1:
            exponent = None
        result = makeTerm(coefficient, variable, exponent)
        return change.done(result)
