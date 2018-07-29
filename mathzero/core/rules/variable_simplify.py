from ..tree_node import LEFT
from ..expressions import (
    AddExpression,
    VariableExpression,
    ConstantExpression,
    MultiplyExpression,
)
from ..util import isAddSubtract, isConstTerm, getTerm, termsAreLike, makeTerm
from .base_rule import BaseRule


class VariableSimplifyRule(BaseRule):
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
        self.l_term = getTerm(node.left)
        if (
            self.l_term == False
            or len(self.l_term.variables) != 1
            or len(self.l_term.coefficients) != 1
        ):
            return False
        self.r_term = getTerm(node.right)
        if (
            self.r_term == False
            or len(self.r_term.variables) != 1
            or len(self.r_term.coefficients) != 1
        ):
            return False
        # The variable of both terms must be the same
        if self.l_term.variables[0] != self.r_term.variables[0]:
            return False
        # TODO: Verify this.
        # Exponents must match with adding two terms? 4x^2 + 5x^2
        if not self.l_term.exponent != self.r_term.exponent and is_add_sub:
            return False
        return True

    def applyTo(self, node):
        change = super().applyTo(node)
        change.saveParent()
        coefficient = self.l_term.coefficients + self.r_term.coefficients
        # The "canApply" verified that left/right variable and exponent match
        variable = self.l_term.variables[0]
        exponent = self.l_term.exponent
        result = makeTerm(coefficient, variable, exponent)
        return change.done(result)

