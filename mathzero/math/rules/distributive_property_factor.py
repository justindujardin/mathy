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
from .base_rule import BaseRule

# ### Distributive Property
# `a(b + c) = ab + ac`
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
    def getName(self):
        return "Distributive Factoring"

    def canApplyTo(self, node):
        if not isAddSubtract(node):
            return False

        self.leftTerm = getTerm(node.left)
        if not self.leftTerm:
            return False

        self.rightTerm = getTerm(node.right)
        if not self.rightTerm:
            return False

        f = factorAddTerms(node)
        if not f:
            return False

        if f.best == 1 and not f.variable and not f.exponent:
            return False

        return True

    def shouldApplyTo(self, node):
        if not super().shouldApplyTo(node):
            return False

        f = factorAddTerms(node)
        if not f.variable:
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
        result = MultiplyExpression(a, inside)

        if leftLink:
            unlink(leftLink)
            leftLink.setRight(result)
            result = leftLink

        change.done(result)
        return change

