from ..tree_node import LEFT
from ..expressions import (
    AddExpression,
    VariableExpression,
    ConstantExpression,
    MultiplyExpression,
    PowerExpression,
)
from ..util import (
    isAddSubtract,
    isConstTerm,
    getTerm,
    termsAreLike,
    makeTerm,
    getTerms,
    isSimpleTerm,
)
from .base_rule import BaseRule
import numpy


class SimplifyComplexTerm(BaseRule):
    """
    Simplify a complex term by combinging its various coefficients, variables, and exponents.

    Example:
        Input = 4x * 8 * 2
        Output = 64x


    TODO: A better rule would only take one step per application, but it's not clear-cut how
          to do that cleanly at the moment. e.g. "4x * 8 * 2" -> "32x * 2" -> "64x"
    """

    def getName(self):
        return "Simplify Complex Term"

    def canApplyTo(self, node):
        if not isinstance(node, MultiplyExpression):
            return False
        term = getTerm(node)
        if term == False or isSimpleTerm(node):
            return False
        return True

    def applyTo(self, node):
        change = super().applyTo(node)
        change.saveParent()
        term = getTerm(node)
        coefficients = term.coefficients
        vars = term.variables

        coefficient = numpy.product(coefficients)
        # If there is a variable, the implicit exponent to be summed is 1, otherwise 0
        implicit_exp = len(vars)
        exponent = term.exponent if term.exponent else implicit_exp
        variable = vars[0] if len(vars) > 0 else None
        # Implicitly everything has an exponent of ^1, so leave it off.
        if exponent is not None and exponent == 1:
            exponent = None
        result = makeTerm(coefficient, variable, exponent)
        return change.done(result)
