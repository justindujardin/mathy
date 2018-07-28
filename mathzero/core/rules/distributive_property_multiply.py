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

# **Distribute across a group**
#
# This handles the `a(b + c)` conversion of the distributive property, which
# distributes `a` across both `b` and `c`.
#
# *note: this is useful because it takes a complex Multiply expression and
# replaces it with two simpler ones.  This can expose terms that can be
# combined for further expression simplification.*
#
#                             +
#         *                  / \
#        / \                /   \
#       /   \              /     \
#      a     +     ->     *       *
#           / \          / \     / \
#          /   \        /   \   /   \
#         b     c      a     b a     c
class DistributiveMultiplyRule(BaseRule):
    def getName(self):
        return "Distributive Multiply"

    def canApplyTo(self, expression):
        if isinstance(expression, MultiplyExpression):
            if expression.right and isinstance(expression.left, AddExpression):
                return True

            if expression.left and isinstance(expression.right, AddExpression):
                return True

            return False

        return False

    def applyTo(self, node):
        change = super().applyTo(node).saveParent()

        if isinstance(node.left, AddExpression):
            a = node.right
            b = node.left.left
            c = node.left.right
        else:
            a = node.left
            b = node.right.left
            c = node.right.right

        a = unlink(a)
        ab = MultiplyExpression(a.clone(), b.clone())
        ac = MultiplyExpression(a.clone(), c.clone())
        result = AddExpression(ab, ac)

        change.done(result)
        return change

