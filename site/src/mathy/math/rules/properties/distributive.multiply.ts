/*
 * decaffeinate suggestions:
 * DS102: Remove unnecessary code created because of implicit returns
 * Full docs: https://github.com/decaffeinate/decaffeinate/blob/master/docs/suggestions.md
 */
// ### Distributive Property
// `a(b + c) = ab + ac`
//
// The distributive property can be used to expand out expressions
// to allow for simplification, as well as to factor out common properties of terms.

// **Distribute across a group**
//
// This handles the `a(b + c)` conversion of the distributive property, which
// distributes `a` across both `b` and `c`.
//
// *note: this is useful because it takes a complex Multiply expression and
// replaces it with two simpler ones.  This can expose terms that can be
// combined for further expression simplification.*
//
//                             +
//         *                  / \
//        / \                /   \
//       /   \              /     \
//      a     +     ->     *       *
//           / \          / \     / \
//          /   \        /   \   /   \
//         b     c      a     b a     c
export class DistributiveMultiplyRule extends BaseRule {
  getName() {
    return 'Distributive Multiply'
  }
  canApplyTo(expression) {
    if (expression instanceof MultiplyExpression) {
      if (expression.right && expression.left instanceof AddExpression) {
        return true
      }
      if (expression.left && expression.right instanceof AddExpression) {
        return true
      }
      return false
    }
    return false
  }
  applyTo(node) {
    let a, b, c
    const change = super.applyTo(node).saveParent()

    if (node.left instanceof AddExpression) {
      a = node.right
      b = node.left.left
      c = node.left.right
    } else {
      a = node.left
      b = node.right.left
      c = node.right.right
    }

    a = unlink(a)
    const ab = new MultiplyExpression(a.clone(), b.clone())
    const ac = new MultiplyExpression(a.clone(), c.clone())
    const result = new AddExpression(ab, ac)

    change.done(result)
    return change
  }
}
