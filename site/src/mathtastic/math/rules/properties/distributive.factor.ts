import { MathExpression } from './../../expressions'
import { BaseRule } from '../../rule'

import { isAddSubtract, getTerm, factorAddTerms, makeTerm, unlink } from '../util'
import { AddExpression, SubtractExpression, MultiplyExpression } from '../../expressions'

// ### Distributive Property
// `a(b + c) = ab + ac`
//
// The distributive property can be used to expand out expressions
// to allow for simplification, as well as to factor out common properties of terms.

// **Factor out a common term**
//
// This handles the `ab + ac` conversion of the distributive property, which factors
// out a common term from the given two addition operands.
//
//           +               *
//          / \             / \
//         /   \           /   \
//        /     \    ->   /     \
//       *       *       a       +
//      / \     / \             / \
//     a   b   a   c           b   c
export class DistributiveFactorOutRule extends BaseRule {
  getName() {
    return 'Distributive Factoring'
  }
  canApplyTo(node: MathExpression) {
    if (!isAddSubtract(node) || isAddSubtract(node.left) || isAddSubtract(node.right)) {
      return false
    }
    const leftTerm = getTerm(node.left)
    if (!leftTerm) {
      return false
    }
    const rightTerm = getTerm(node.right)
    if (!rightTerm) {
      return false
    }
    // Don't try factoring out terms with multiple variables, e.g "(4z + 84xz)"
    if (leftTerm.variables.length > 1 || rightTerm.variables.length > 1) {
      return false
    }
    const f = factorAddTerms(node)
    if (!f) {
      return false
    }
    if (f.best === 1 && !f.variable && !f.exponent) {
      return false
    }
    return true
  }
  applyTo(node: MathExpression) {
    let leftLink
    const change = super.applyTo(node).saveParent()
    if (node.left && isAddSubtract(node.left)) {
      leftLink = node.left.clone()
    }

    const factors = factorAddTerms(node)
    if (factors === false) {
      throw new Error('invalid rule state')
    }

    const a = makeTerm(factors.best, factors.variable, factors.exponent)
    const b = makeTerm(factors.left, factors.leftVariable, factors.leftExponent)
    const c = makeTerm(factors.right, factors.rightVariable, factors.rightExponent)

    const inside =
      node instanceof AddExpression ? new AddExpression(b, c) : new SubtractExpression(b, c)
    let result = new MultiplyExpression(a, inside)

    if (leftLink) {
      unlink(leftLink)
      leftLink.setRight(result)
      result = leftLink
    }

    change.done(result)
    return change
  }
}
