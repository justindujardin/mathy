from .expressions import (
    ConstantExpression,
    VariableExpression,
    MultiplyExpression,
    PowerExpression,
    AddExpression,
    SubtractExpression,
    BinaryExpression,
    NegateExpression,
    MathExpression,
)

import numpy
import math

# Unlink an expression from it's parent.
#
# 1. Clear expression references in `parent`
# 2. Clear `parent` in expression
def unlink(node):
    if not node:
        return None

    if node.parent:
        if node == node.parent.left:
            node.parent.setLeft(None)

        if node == node.parent.right:
            node.parent.setRight(None)

    node.parent = None
    return node


# Build a verbose factor dictionary.
#
# This builds a dictionary of factors for a given value that
# contains both arrangements of terms so that all factors are
# accessible by key.  That is, factoring 2 would return
#      result =
#        1 : 2
#        2 : 1
def factor(value):
    if value == 0 or math.isnan(value):
        return []

    sqrt = numpy.sqrt(value)
    if math.isnan(sqrt):
        return []

    flip = value < 0
    sqrt = int(sqrt + 1)
    factors = {1: value}
    factors[value] = 1

    for i in range(2, sqrt):
        if value % i == 0:
            one = i
            two = value / i
            factors[one] = two
            factors[two] = one
    return factors


def isAddSubtract(node):
    return isinstance(node, AddExpression) or isinstance(node, SubtractExpression)


class FactorResult:
    def __init__(self):
        self.best = -1
        self.left = -1
        self.right = -1
        self.allLeft = []
        self.allRight = []
        self.variable = None
        self.exponent = None
        self.leftExponent = None
        self.rightExponent = None
        self.leftVariable = None
        self.rightVariable = None


def factorAddTerms(node):
    if not isAddSubtract(node):
        raise Exception("Cannot factor non add/subtract node")

    lTerm = getTerm(node.left)
    rTerm = getTerm(node.right)
    if not lTerm or not rTerm:
        raise Exception("Complex or unidentifiable term/s in {}".format(node))

    # TODO: Skipping complex terms with multiple coefficients for now.
    if lTerm.coefficients and len(lTerm.coefficients) > 1:
        return False

    if rTerm.coefficients and len(rTerm.coefficients) > 1:
        return False

    # Common coefficients
    lCoefficients = factor(lTerm.coefficients[0] if len(lTerm.coefficients) > 0 else 1)
    rCoefficients = factor(rTerm.coefficients[0] if len(rTerm.coefficients) > 0 else 1)
    common = [k for k in rCoefficients if k in lCoefficients]
    if len(common) == 0:
        return False
    best = numpy.max(common)
    result = FactorResult()
    result.best = best
    result.left = lCoefficients[best]
    result.right = rCoefficients[best]
    result.allLeft = lCoefficients
    result.allRight = rCoefficients

    # Common variables and powers
    commonExp = lTerm.exponent and rTerm.exponent and lTerm.exponent == rTerm.exponent
    expMatch = False if (lTerm.exponent or rTerm.exponent) and not commonExp else True
    hasLeft = len(lTerm.variables) > 0
    hasRight = len(rTerm.variables) > 0
    if hasLeft and hasRight and lTerm.variables[0] == rTerm.variables[0] and expMatch:
        result.variable = lTerm.variables[0]
        result.exponent = lTerm.exponent

    if lTerm.exponent and lTerm.exponent != result.exponent:
        result.leftExponent = lTerm.exponent

    if rTerm.exponent and rTerm.exponent != result.exponent:
        result.rightExponent = rTerm.exponent

    if hasLeft and lTerm.variables[0] != result.variable:
        result.leftVariable = lTerm.variables[0]

    if hasRight and rTerm.variables[0] != result.variable:
        result.rightVariable = rTerm.variables[0]

    return result


# Create a term node hierarchy from a given set of
# term parameters.  This takes into account removing
# implicit coefficients of 1 where possible.
def makeTerm(coefficient, variable, exponent):
    constExp = ConstantExpression(coefficient)
    if not variable and not exponent:
        return constExp

    varExp = VariableExpression(variable)
    if coefficient == 1 and not exponent:
        return varExp

    multExp = MultiplyExpression(constExp, varExp)
    if not exponent:
        return multExp

    expConstExp = ConstantExpression(exponent)
    if coefficient == 1:
        return PowerExpression(varExp, expConstExp)

    return PowerExpression(multExp, expConstExp)


class TermResult:
    def __init__(self):
        self.coefficients = []
        self.variables = []
        self.exponent = None
        self.node_coefficients = []
        self.node_variables = []
        self.node_exponent = None


# Extract term information from the given node
#
def getTerm(node):
    result = TermResult()
    # Constant with add/sub parent should be OKAY.
    if isinstance(node, ConstantExpression):
        if not node.parent or (node.parent and isAddSubtract(node.parent)):
            result.coefficients = [node.value]
            result.node_coefficients = [node]
            return result

    # Variable with add/sub parent should be OKAY.
    if isinstance(node, VariableExpression):
        if not node.parent or (node.parent and isAddSubtract(node.parent)):
            result.variables = [node.identifier]
            result.node_variables = [node]
            return result

    # TODO: Comment resolution on whether +- is OKAY, and if not, why it breaks down.
    if not isAddSubtract(node):
        if (
            len(node.findByType(AddExpression)) > 0
            or len(node.findByType(SubtractExpression)) > 0
        ):
            return False

    # If another add is found on the left side of this node, and the right node
    # is _NOT_ a leaf, we cannot extract a term.  If it is a leaf, the term should be
    # just the right node.
    if node.left and len(node.left.findByType(AddExpression)) > 0:
        if node.right and not node.right.isLeaf():
            return False

    if node.right and len(node.right.findByType(AddExpression)) > 0:
        return False

    exponents = node.findByType(PowerExpression)
    if len(exponents) > 0:
        # Supports only single exponents in terms
        if len(exponents) != 1:
            return False

        exponent = exponents[0]
        if not isinstance(exponent.right, ConstantExpression):
            raise Exception("getTerm supports constant term powers")

        result.exponent = exponent.right.value
        result.node_exponent = exponent

    variables = node.findByType(VariableExpression)
    if len(variables) > 0:
        result.variables = [v.identifier for v in variables]
        result.node_variables = variables

    def filter_coefficients(n):
        if not n.parent or n.parent == node.parent:
            return True

        if isinstance(n.parent, BinaryExpression) and not isinstance(
            n.parent, MultiplyExpression
        ):
            return False
        return True

    coefficients = node.findByType(ConstantExpression)
    coefficients = [c for c in coefficients if filter_coefficients(c)]
    if len(coefficients) > 0:

        def resolve_coefficients(c):
            value = c.value
            if isinstance(c.parent, NegateExpression):
                value = value * -1

            return value

        result.coefficients = [resolve_coefficients(c) for c in coefficients]
        result.node_coefficients = coefficients

    empty = (
        len(result.variables) == 0
        and len(result.coefficients) == 0
        and result.exponent is None
    )
    if empty:
        return False

    # consistently return an empty coefficients/variables array in case none exist.
    # this ensures that you can always reference coefficients[0] or variables[0] and
    # check that for truthiness, rather than having to check that the object property
    # `coefficients` or `variables` is not None and also the truthiness of index 0.
    if not result.coefficients:
        result.coefficients = []

    if not result.variables:
        result.variables = []
    return result


def getTerms(node):
    terms = []
    while node:
        if not isAddSubtract(node):
            terms.append(node)
            break
        else:
            if node.right:
                terms.append(node.right)

        node = node.left
    return terms


def termsAreLike(one, two):
    """
    @param {Object|MathExpression} one The first term {@link #getTerm}
    @param {Object|MathExpression} two The second term {@link #getTerm}
    @returns {Boolean} Whether the terms are like or not.
    """
    # Both must be valid terms
    if not one or not two:
        return False

    # Extract terms from MathExpressions if need be
    if isinstance(one, MathExpression):
        one = getTerm(one)

    if isinstance(two, MathExpression):
        two = getTerm(two)

    # Both must have variables, and they must match exactly.
    if not one.variables or not two.variables:
        return False

    if len(one.variables) == 0 or len(two.variables) == 0:
        return False

    if len(one.variables) != len(two.variables):
        return False

    invalid = len([False for v in one.variables if v not in two.variables]) > 0
    if invalid:
        return False

    # Also, the exponents must match
    if one.exponent != two.exponent:
        return False

    # Same variables, and exponents.  Good to go.
    return True


# Negate helper

# `l - r = l + (-r)`
#
#             -                  +
#            / \                / \
#           /   \     ->       /   \
#          /     \            /     \
#         *       2          *       -
#        / \                / \       \
#       4   x              4   x       2
def negate(node):
    save = node.parent
    saveSide = save.getSide(node) if save != None else None
    unlink(node)
    newNode = AddExpression(node.left, NegateExpression(node.right))
    if save != None:
        save.setSide(newNode, saveSide)

    return newNode


# Determine if an expression represents a constant term
def isConstTerm(node):
    if isinstance(node, ConstantExpression):
        return True

    if isinstance(node, NegateExpression) and isConstTerm(node.child):
        return True

    return False


def getTermConst(node):
    if isinstance(node, ConstantExpression):
        return node.value

    if isinstance(node.left, ConstantExpression):
        return node.left.value

    if isinstance(node.right, ConstantExpression):
        return node.right.value

    raise Exception("Unable to determine coefficient for expression")

