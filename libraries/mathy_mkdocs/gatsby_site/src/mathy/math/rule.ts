import { MathExpression } from './expressions'
import { STOP } from './treeNode'

export abstract class BaseRule {
  name = 'Abstract Base Rule'
  public findNode(expression: MathExpression) {
    let result: MathExpression | null = null
    expression.visitPreorder((node: MathExpression) => {
      if (this.canApplyTo(node)) {
        result = node
      }
      if (result !== null) {
        return STOP
      }
    })
    return result
  }

  /**
   * Find all nodes in an expression that can have this rule applied to them.
   * Each node is marked with it's token index in the expression, according to
   * the visit strategy, and stored as `node.token` starting with index 0
   * @param expression
   */
  public findNodes(expression: MathExpression) {
    const nodes: MathExpression[] = []
    let index = 0
    expression.visitPreorder((node: MathExpression) => {
      node.token = index
      if (this.canApplyTo(node)) {
        nodes.push(node)
      }
      index += 1
    })
    return nodes
  }
  public canApplyTo(node: MathExpression) {
    return false
  }
  public applyTo(node: MathExpression) {
    if (!this.canApplyTo(node)) {
      throw new Error(`Cannot apply ${this.name} to ${node}`)
    }
    return new ExpressionChangeRule(this, node)
  }
}

// Basic description of a change to an expression tree
export class ExpressionChangeRule<T extends BaseRule = BaseRule> {
  public result: MathExpression | null = null
  private _saveParent: MathExpression | null = null
  private _saveSide: string | null = null
  constructor(public rule: T, public node?: MathExpression) {}

  public saveParent(parent: MathExpression | null = null, side: string | null = null) {
    this._saveParent = parent
    if (this.node && parent === null) {
      this._saveParent = this.node
    }
    if (this._saveParent) {
      this._saveSide = side || this._saveParent.getSide(this.node)
    }
    return this
  }

  public done(node: MathExpression) {
    if (this._saveParent) {
      this._saveParent.setSide(node, this._saveSide)
    }
    this.result = node
    return this
  }
}
