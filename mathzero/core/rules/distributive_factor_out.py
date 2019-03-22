from ..expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
    SubtractExpression,
)
from ..util import (
    isAddSubtract,
    isConstTerm,
    getTerm,
    termsAreLike,
    unlink,
    factorAddTerms,
    makeTerm,
)
from ..rule import BaseRule

# ### Distributive Property
# `ab + ac = a(b + c)`
#
# The distributive property can be used to expand out expressions
# to allow for simplification, as well as to factor out common properties of terms.

# **Factor out a common term**
#
# This handles the `ab + ac` conversion of the distributive property, which factors
# out a common term from the given two addition operands.
#
#           +               *
#          / \             / \
#         /   \           /   \
#        /     \    ->   /     \
#       *       *       a       +
#      / \     / \             / \
#     a   b   a   c           b   c
class DistributiveFactorOutRule(BaseRule):
    @property
    def name(self):
        return "Distributive Factoring"

    @property
    def code(self):
        return "DF"

    def canApplyTo(self, node):
        if (
            not isAddSubtract(node)
            or isAddSubtract(node.left)
            or isAddSubtract(node.right)
        ):
            return False

        leftTerm = getTerm(node.left)
        if not leftTerm:
            return False

        rightTerm = getTerm(node.right)
        if not rightTerm:
            return False

        # Don't try factoring out terms with multiple variables, e.g "(4z + 84xz)"
        if len(leftTerm.variables) > 1 or len(rightTerm.variables) > 1:
            return False

        f = factorAddTerms(node)
        if not f:
            return False

        if f.best == 1 and not f.variable and not f.exponent:
            return False

        return True

    def applyTo(self, node):
        leftLink = None
        change = super().applyTo(node).saveParent()
        if isAddSubtract(node.left):
            leftLink = node.left.clone()

        factors = factorAddTerms(node)

        a = makeTerm(factors.best, factors.variable, factors.exponent)
        b = makeTerm(factors.left, factors.leftVariable, factors.leftExponent)
        c = makeTerm(factors.right, factors.rightVariable, factors.rightExponent)

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

        if leftLink:
            unlink(leftLink)
            leftLink.setRight(result)
            result = leftLink

        change.done(result)
        return change
