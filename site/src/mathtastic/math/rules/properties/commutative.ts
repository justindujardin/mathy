import { MathExpression } from './../../expressions'
import { BaseRule } from '../../rule'

import { AddExpression, MultiplyExpression } from '../../expressions'

/*
 * decaffeinate suggestions:
 * DS102: Remove unnecessary code created because of implicit returns
 * Full docs: https://github.com/decaffeinate/decaffeinate/blob/master/docs/suggestions.md
 */
// ### Commutative Property
//
//
// For Addition: `a + b = b + a`
//
//             +                  +
//            / \                / \
//           /   \     ->       /   \
//          /     \            /     \
//         a       b          b       a
//
// For Multiplication: `ab = ba`
//
//             *                  *
//            / \                / \
//           /   \     ->       /   \
//          /     \            /     \
//         a       b          b       a
//
export class CommutativeSwapRule extends BaseRule {
  getName() {
    return 'Commutative Move'
  }
  canApplyTo(node: MathExpression) {
    // Must be an add/multiply
    return node instanceof AddExpression || node instanceof MultiplyExpression
  }
  applyTo(node: MathExpression) {
    const change = super.applyTo(node)
    const a = node.left
    const b = node.right

    node.setRight(a)
    node.setLeft(b)

    change.done(node)
    return change
  }
}
