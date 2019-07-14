import { MathExpression } from '../../expressions'
import { AddExpression, MultiplyExpression } from '../../expressions'
import { BaseRule, ExpressionChangeRule } from '../../rule'

// ### Associative Property
//
// Addition: `(a + b) + c = a + (b + c)`
//
//         (y) +            + (x)
//            / \          / \
//           /   \        /   \
//      (x) +     c  ->  a     + (y)
//         / \                / \
//        /   \              /   \
//       a     b            b     c
//
// Multiplication: `(ab)c = a(bc)`
//
//         (x) *            * (y)
//            / \          / \
//           /   \        /   \
//      (y) *     c  <-  a     * (x)
//         / \                / \
//        /   \              /   \
//       a     b            b     c
//
export class AssociativeSwapRule extends BaseRule {
  public getName() {
    return 'Associative Group'
  }

  public canApplyTo(node: MathExpression): boolean {
    if (node.parent instanceof AddExpression && node instanceof AddExpression) {
      return true
    }
    if (node.parent instanceof MultiplyExpression && node instanceof MultiplyExpression) {
      return true
    }
    return false
  }
  public applyTo(node: MathExpression): ExpressionChangeRule<this> {
    const change = super.applyTo(node)
    node.rotate()
    change.done(node)
    return change
  }
}
