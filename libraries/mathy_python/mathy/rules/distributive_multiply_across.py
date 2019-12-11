from ..core.expressions import (
    AddExpression,
    MultiplyExpression,
    ConstantExpression,
    VariableExpression,
    PowerExpression,
)
from ..core.rule import BaseRule
from ..util import unlink


class DistributiveMultiplyRule(BaseRule):
    r"""
    Distributive Property
    `a(b + c) = ab + ac`

    The distributive property can be used to expand out expressions
    to allow for simplification, as well as to factor out common properties of terms.

    **Distribute across a group**

    This handles the `a(b + c)` conversion of the distributive property, which
    distributes `a` across both `b` and `c`.

    *note: this is useful because it takes a complex Multiply expression and
    replaces it with two simpler ones.  This can expose terms that can be
    combined for further expression simplification.*

                                 +
             *                  / \
            / \                /   \
           /   \              /     \
          a     +     ->     *       *
               / \          / \     / \
              /   \        /   \   /   \
             b     c      a     b a     c
    """

    @property
    def name(self):
        return "Distributive Multiply"

    @property
    def code(self):
        return "DM"

    def can_apply_to(self, expression):
        if isinstance(expression, MultiplyExpression):
            if expression.right and isinstance(expression.left, AddExpression):
                return True

            if expression.left and isinstance(expression.right, AddExpression):
                return True

            return False

        return False

    def apply_to(self, node):
        change = super().apply_to(node).save_parent()

        if isinstance(node.left, AddExpression):
            a = node.right
            b = node.left.left
            c = node.left.right
        else:
            a = node.left
            b = node.right.left
            c = node.right.right

        a = unlink(a)

        # If the operands for either multiplication can be expressed
        # in a "natural" order, do so like a human would.
        a_exp_var = (
            isinstance(a, PowerExpression)
            and isinstance(a.right, VariableExpression)
            or isinstance(a.left, VariableExpression)
        )
        a_var: bool = a_exp_var or isinstance(a, VariableExpression)
        b_const: bool = isinstance(b, ConstantExpression)
        c_const: bool = isinstance(c, ConstantExpression)
        if a_var and b_const:
            ab = MultiplyExpression(b.clone(), a.clone())
        else:
            ab = MultiplyExpression(a.clone(), b.clone())

        if a_var and c_const:
            ac = MultiplyExpression(c.clone(), a.clone())
        else:
            ac = MultiplyExpression(a.clone(), c.clone())
        result = AddExpression(ab, ac)

        [n.set_changed() for n in [node, a, ab, ac, result]]

        change.done(result)
        return change
