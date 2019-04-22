import { BinaryTreeNode, STOP } from './treeNode'
import { uuidv4 } from './util'
import _ from 'lodash'

// ## Math Expression

// ## Constants

export const OOO_FUNCTION = 4
export const OOO_PARENS = 3
export const OOO_EXPONENT = 2
export const OOO_MULTDIV = 1
export const OOO_ADDSUB = 0
export const OOO_INVALID = -1

/**
 * Variables and values for the context that should be used while
 * evaluating an expression, e.g. {x: 3, y:1.2}
 */
export interface IEvaluationContext {
  [key: string]: any
}

/** Interface for holding state data for tree cloning @private  */
interface IRootCloneState {
  targetClone: MathExpression | null
  clonedNode: MathExpression | null
}

/** * A Basic MathExpression node */
export class MathExpression extends BinaryTreeNode {
  public left: MathExpression | null = null
  public right: MathExpression | null = null
  public parent: MathExpression | null = null
  /**
   * The token index of this node in its parent expression tree when
   * visited pre order. This is only used by the rules system so it's
   * usually just the default -1 value.
   */
  public token: number = -1
  public classes: string[] = []
  private _rootClone: IRootCloneState | null = null
  constructor(public id = `mn-${uuidv4()}`) {
    super()
    this.classes.push(this.id)
  }
  /**
   * Evaluate the expression, resolving all variables to constant values
   */
  public evaluate(_context: IEvaluationContext): number {
    return 0.0
  }
  /**
   * Differentiate the expression by a given variable
   * @abstract
   */
  public differentiate(_byVar: string): MathExpression {
    throw new Error('cannot differentiate an abstract MathExpression node')
  }

  /**
   * Associate a class name with an expression.  This class name will be tagged on nodes
   * when the expression is converted to a capable output format.  See {@link #getMathML}.
   */
  public addClass(classes: string[]) {
    this.classes = _.union(_.isArray(classes) ? classes : [classes], this.classes)
  }
  /**
   * Find an expression in this tree by type.
   * @param {Function} instanceType The type to check for instances of
   * @returns {Array} Array of {@link MathExpression} that are of the given type.
   */
  public findByType<T extends MathExpression = MathExpression>(instanceType: Function): T[] {
    const results: T[] = []
    this.visitInorder((node: T) => {
      if (node instanceof instanceType) {
        return results.push(node)
      }
    })
    return results
  }
  /**
   * Find an expression by its unique ID.
   * @returns {MathExpression|null} The node.
   */
  public findById(id: string): MathExpression | null {
    let result = null
    this.visitInorder(function(node: MathExpression) {
      if (node.id === id) {
        result = node
        return STOP
      }
    })
    return result
  }

  /** Convert this single node into MathML. */
  public toMathML() {
    return ''
  }
  /** Convert this expression into a MathML container. */
  public getMathML() {
    return ["<math xmlns='http://www.w3.org/1998/Math/MathML'>", this.toMathML(), '</math>'].join(
      '\n'
    )
  }

  /**
   * Make an ML tag for the given content, respecting the node's
   * given classes.
   * @param {String} tag The ML tag name to create.
   * @param {String} content The ML content to place inside of the tag.
   * @param {Array} classes An array of classes to attach to this tag.
   */
  public makeMLTag(tag: string, content: string | number, classes: string[] = []) {
    const attr = classes.length ? ` class='${classes.join(' ')}'` : ''
    return `<${tag}${attr}>${content}</${tag}>`
  }

  /**
   * Like the clone method, but also clones the parent hierarchy up to
   * the node root.  This is useful when you want to clone a subtree and still
   * maintain the overall hierarchy.
   * @param {MathExpression} [node=this] The node to clone.
   * @returns {MathExpression} The cloned node.
   */
  public rootClone(node: MathExpression = this) {
    this._rootClone = {
      clonedNode: null,
      targetClone: node
    }
    let result = node.getRoot().clone()
    if (!this._rootClone.clonedNode) {
      throw new Error('cloning root hierarchy did not clone this node')
    }
    result = this._rootClone.clonedNode
    this._rootClone = null
    return result
  }

  /**
   * A specialization of the clone method that can track and report a cloned subtree
   * node.  See {@link #rootClone} for more details.
   */
  public clone() {
    const result = super.clone()
    if (this._rootClone && this._rootClone.targetClone === this) {
      this._rootClone.clonedNode = result
    }
    return result
  }

  /**
   * Determine if this is a sub-term, meaning it has
   * a parent that is also a term, in the expression.
   *
   * This indicates that a term has limited mobility,
   * and cannot be freely moved around the entire expression.
   */
  public isSubTerm() {
    let node = this.parent
    while (node) {
      if (node.getTerm() !== false) {
        return true
      }
      node = node.parent
    }
    return false
  }
  /**
   * Get the term that this node belongs to.
   * @returns {boolean|MathExpression}
   */
  public getTerm(): boolean | MathExpression | null {
    if (this instanceof AddExpression || this instanceof SubtractExpression) {
      return false
    }
    // tslint:disable-next-line:no-this-assignment
    let node: MathExpression | null = this
    while (node && node.parent) {
      // If there's a multiplication it's a term.  It can be part of a larger term,
      // but technically it's still a term.  Identify it as such.
      if (node instanceof MultiplyExpression) {
        return node
      }
      // If we have an add/subtract parent, yep, definitely a term.
      if (node.parent instanceof AddExpression || node.parent instanceof SubtractExpression) {
        return node
      }
      node = node.parent
    }
    return node
  }

  /**
   * Get any terms that are children of this node.
   * @returns {Array} MathExpression
   */
  public getTerms() {
    const terms: MathExpression[] = []
    this.visitPreorder(function(node: MathExpression) {
      // If the parent is not an Add/Sub/Equal, not a term.
      if (
        !(node.parent instanceof AddExpression) &&
        !(node.parent instanceof SubtractExpression) &&
        !(node.parent instanceof EqualExpression)
      ) {
        return
      }
      // If the node is an Add/Sub/Equal, not a term.
      if (
        node instanceof AddExpression ||
        node instanceof SubtractExpression ||
        node instanceof EqualExpression
      ) {
        return
      }
      // Otherwise, looks good.
      return terms.push(node)
    })
    return terms
  }

  /**
   * Return a number representing the order of operations priority
   * of this node.  This can be used to check if a node is `locked`
   * with respect to another node, i.e. the other node must be resolved
   * first during evaluation because of it's priority.
   */
  public getPriority(): number {
    let priority = OOO_INVALID
    if (this instanceof AddExpression || this instanceof SubtractExpression) {
      priority = OOO_ADDSUB
    }
    if (this instanceof MultiplyExpression || this instanceof DivideExpression) {
      priority = OOO_MULTDIV
    }
    if (this instanceof PowerExpression) {
      priority = OOO_EXPONENT
    }
    if (this instanceof FunctionExpression) {
      priority = OOO_FUNCTION
    }
    return priority
  }
}

/** An expression that operates on one sub-expression */
export class UnaryExpression extends MathExpression {
  constructor(public child: MathExpression, public operatorleft = true) {
    super()
    this.setChild(child)
  }
  public setChild(child: MathExpression) {
    if (this.operatorleft) {
      return this.setLeft(child)
    }
    return this.setRight(child)
  }
  public getChild() {
    if (this.operatorleft) {
      return this.left
    }
    return this.right
  }
  public evaluate(context: IEvaluationContext): number {
    const child = this.getChild()
    if (child === null) {
      return 0.0
    }
    return this.operate(child.evaluate(context))
  }
  /** @abstract */
  public operate(_value: number): number {
    throw 'Must be implemented in subclass'
  }
}

// ### Negation

/** Negate an expression, e.g. `4` becomes `-4` */
export class NegateExpression extends UnaryExpression {
  public getName() {
    return '-'
  }
  public operate(value: number) {
    return -value
  }
  public toString() {
    return `-${this.getChild()}`
  }
  /**
   * <pre>
   *           f(x) = -g(x);
   *      d( f(x) ) = -( d( g(x) ) );
   * </pre>
   */
  public differentiate(byVar: string): MathExpression {
    if (!this.child) {
      throw new Error('invalid child')
    }
    return new NegateExpression(this.child.differentiate(byVar))
  }
}

// ### Function

/**
 * A Specialized UnaryExpression that is used for functions.  The function name in
 * text (used by the parser and tokenizer) is derived from the getName() method on
 * the class.
 */
export class FunctionExpression extends UnaryExpression {
  public getName() {
    return ''
  }
  public toString() {
    const child = this.getChild()
    if (child) {
      return `${this.getName()}(${child})`
    }
    return `${this.getName()}`
  }
}

// ## Binary Expressions

/** An expression that operates on two sub-expressions */
export class BinaryExpression extends MathExpression {
  constructor(left: MathExpression, right: MathExpression) {
    super()
    this.setLeft(left)
    this.setRight(right)
  }
  public evaluate(context: IEvaluationContext): number {
    if (!this.left || !this.right) {
      throw new Error(
        `cannot operate on invalid left(${this.left}) or right(${this.right}) expression`
      )
    }
    const leftResult = this.left.evaluate(context)
    const rightResult = this.right.evaluate(context)
    if (!leftResult || !rightResult) {
      throw new Error(`invalid left(${leftResult}) or right(${rightResult}) result`)
    }
    return this.operate(leftResult, rightResult)
  }
  /** @abstract */
  public getName(): string {
    throw new Error('Must be implemented in subclass')
  }
  public getMLName() {
    return this.getName()
  }
  /** @abstract */
  public operate(_one: number, _two: number): number {
    throw new Error('Must be implemented in subclass')
  }

  public leftParenthesis(): boolean {
    if (!this.left) {
      return false
    }
    const leftChildBinary = this.left && this.left instanceof BinaryExpression
    return leftChildBinary && this.left && this.left.getPriority() < this.getPriority()
  }

  public rightParenthesis(): boolean {
    if (!this.right) {
      return false
    }
    const rightChildBinary = this.right instanceof BinaryExpression
    return rightChildBinary && this.right.getPriority() < this.getPriority()
  }

  public toString() {
    if (this.rightParenthesis()) {
      return `${this.left} ${this.getName()} (${this.right})`
    }
    if (this.leftParenthesis()) {
      return `(${this.left}) ${this.getName()} ${this.right}`
    }
    return `${this.left} ${this.getName()} ${this.right}`
  }

  public toMathML() {
    const rightML = this.right ? this.right.toMathML() : ''
    const leftML = this.left ? this.left.toMathML() : ''
    const opML = this.makeMLTag('mo', this.getMLName())
    if (this.rightParenthesis()) {
      return this.makeMLTag('mrow', `${leftML}${opML}<mo>(</mo>${rightML}<mo>)</mo>`, this.classes)
    }
    if (this.leftParenthesis()) {
      return this.makeMLTag('mrow', `<mo>(</mo>${leftML}<mo>)</mo>${opML}${rightML}`, this.classes)
    }
    return this.makeMLTag('mrow', `${leftML}${opML}${rightML}`, this.classes)
  }
}

/** Evaluate equality of two expressions */
export class EqualExpression extends BinaryExpression {
  public getName() {
    return '='
  }
  /**
   * @abstract
   * This is where assignment of context variables might make sense.  But context is not
   * present in the expression's `operate` method.  TODO: Investigate this thoroughly.
   */
  public operate(_one: number, _two: number): number {
    throw new Error('Unsupported operation. Euqality has no operation to perform.')
  }
}

/** Add one and two */
export class AddExpression extends BinaryExpression {
  public getName() {
    return '+'
  }
  public operate(one: number, two: number): number {
    return one + two
  }

  //           f(x) = g(x) + h(x);
  //      d( f(x) ) = d( g(x) ) + d( h(x) );
  //          f'(x) = g'(x) + h'(x);
  public differentiate(byVar: string): MathExpression {
    if (!this.left || !this.right) {
      throw new Error('invalid left/right children')
    }
    return new AddExpression(this.left.differentiate(byVar), this.right.differentiate(byVar))
  }
}

/** Subtract one from two */
export class SubtractExpression extends BinaryExpression {
  public getName() {
    return '-'
  }
  public operate(one: number, two: number): number {
    return one - two
  }
  //           f(x) = g(x) - h(x);
  //      d( f(x) ) = d( g(x) ) - d( h(x) );
  //          f'(x) = g'(x) - h'(x);
  public differentiate(byVar: string): MathExpression {
    if (!this.left || !this.right) {
      throw new Error('invalid left/right children')
    }
    return new AddExpression(this.left.differentiate(byVar), this.right.differentiate(byVar))
  }
}

/** Multiply one and two */
export class MultiplyExpression extends BinaryExpression {
  public getName() {
    return '*'
  }
  public getMLName() {
    return '&#183;'
  }
  public operate(one: number, two: number): number {
    return one * two
  }
  //      f(x) = g(x)*h(x);
  //     f'(x) = g(x)*h'(x) + g'(x)*h(x);
  public differentiate(byVar: string): MathExpression {
    if (!this.left || !this.right) {
      throw new Error('invalid left/right children')
    }

    return new AddExpression(
      new MultiplyExpression(this.left, this.right.differentiate(byVar)),
      new MultiplyExpression(this.left.differentiate(byVar), this.right)
    )
  }
  // Multiplication special cases constant*variable case to output as, e.g. "4x"
  // instead of "4 * x"
  public toString(): string {
    if (this.left instanceof ConstantExpression) {
      if (this.right instanceof VariableExpression || this.right instanceof PowerExpression) {
        return `${this.left}${this.right}`
      }
    }
    return super.toString()
  }

  public toMathML() {
    if (!this.left || !this.right) {
      throw new Error('invalid left/right children')
    }
    const rightML = this.right.toMathML()
    const leftML = this.left.toMathML()
    if (this.left instanceof ConstantExpression) {
      if (this.right instanceof VariableExpression || this.right instanceof PowerExpression) {
        return `${leftML}${rightML}`
      }
    }
    return super.toMathML()
  }
}

/** Divide one by two */
export class DivideExpression extends BinaryExpression {
  public getName() {
    return '/'
  }
  public getMLName() {
    return '&#247;'
  }
  // toMathML:() -> "<mfrac>#{@left.toMathML()}#{@right.toMathML()}</mfrac>"
  public operate(one: number, two: number): number {
    if (two === 0) {
      return NaN
    }
    return one / two
  }
  //       f(x) = g(x)/h(x)
  //      f'(x) = ( g'(x)*h(x) - g(x)*h'(x) ) / ( h(x)^2 )
  public differentiate(byVar: string) {
    if (!this.left || !this.right) {
      throw new Error('invalid left/right children')
    }
    const gprimeh = new MultiplyExpression(this.left.differentiate(byVar), this.right)
    const ghprime = new MultiplyExpression(this.left, this.right.differentiate(byVar))
    const hsquare = new PowerExpression(this.right, new ConstantExpression(2))
    return new DivideExpression(new SubtractExpression(gprimeh, ghprime), hsquare)
  }
}

/** Raise one to the power of two */
export class PowerExpression extends BinaryExpression {
  public getName() {
    return '^'
  }
  public toMathML() {
    if (!this.right || !this.left) {
      throw new Error('invalid left/right children')
    }
    const rightML = this.right.toMathML()
    let leftML = this.left.toMathML()
    // if left is mult, enclose only right in msup
    if (this.left instanceof MultiplyExpression) {
      leftML = this.makeMLTag('mrow', leftML, this.classes)
    }
    return this.makeMLTag('msup', `${leftML}${rightML}`, this.classes)
  }
  public operate(one: number, two: number) {
    return Math.pow(one, two)
  }
  /** @abstract */
  public differentiate(_byVar: string): MathExpression {
    throw new Error('Unimplemented')
  }
  public toString() {
    return `${this.left}${this.getName()}${this.right}`
  }
}

/** Constant Expression */
export class ConstantExpression extends MathExpression {
  constructor(public value: number) {
    super()
  }
  public clone() {
    const result = super.clone()
    result.value = this.value
    return result
  }
  public evaluate(_context: IEvaluationContext) {
    return this.value
  }
  public toString() {
    return `${this.value}`
  }
  public toJSON() {
    const result = super.toJSON()
    result.name = this.value
    return result
  }
  public toMathML() {
    return this.makeMLTag('mn', this.value, this.classes)
  }
  /**
   * Differentiation of a constant yields 0
   * <pre>
   *           f(x) = c
   *      d( f(x) ) = c * d( c )
   *         d( c ) = 0
   *          f'(x) = 0
   * </pre>
   */
  public differentiate(_byVar: string) {
    return new ConstantExpression(0.0)
  }
}

/** Variable Expression */
export class VariableExpression extends MathExpression {
  constructor(public identifier: string) {
    super()
  }
  public clone() {
    const result = super.clone()
    result.identifier = this.identifier
    return result
  }
  public toString() {
    if (this.identifier === undefined) {
      return ''
    }
    return `${this.identifier}`
  }
  public toMathML() {
    if (this.identifier === undefined) {
      return ''
    }
    return this.makeMLTag('mi', this.identifier)
  }
  public toJSON() {
    const result = super.toJSON()
    result.name = this.identifier
    return result
  }
  public evaluate(context: IEvaluationContext) {
    if (context && context[this.identifier]) {
      return context[this.identifier]
    }
    throw new Error(`cannot evaluate statement with undefined variable: ${this.identifier}`)
  }
  public differentiate(byVar: string) {
    // Differentiating by this variable yields 1
    //
    //          f(x) = x
    //     d( f(x) ) = 1 * d( x )
    //        d( x ) = 1
    //         f'(x) = 1
    if (byVar === this.identifier) {
      return new ConstantExpression(1)
    }
    // Differentiating by any other variable yields 0
    //
    //          f(x) = c
    //     d( f(x) ) = c * d( c )
    //        d( c ) = 0
    //         f'(x) = 0
    return new ConstantExpression(0)
  }
}

/**
 * Absolute Value.
 *
 * Evaluates the absolute value of an expression.
 */
export class AbsExpression extends FunctionExpression {
  public getName() {
    return 'abs'
  }
  public operate(value: number) {
    return Math.abs(value)
  }
  //        f(x)   = abs( g(x) );
  //     d( f(x) ) = sgn( g(x) ) * d( g(x) );
  public differentiate(byVar: string) {
    return new MultiplyExpression(new SgnExpression(this.child), this.child.differentiate(byVar))
  }
}

/**
 * Evaluates the sign of an expression.
 * @class
 * @extends {FunctionExpression}
 */
export class SgnExpression extends FunctionExpression {
  public getName() {
    return 'sgn'
  }
  /**
   * Determine the sign of an value
   * @returns {Number} -1 if negative, 1 if positive, 0 if 0
   */
  public operate(value: number): number {
    if (value < 0) {
      return -1
    }
    if (value > 0) {
      return 1
    }
    return 0
  }
  /**
   * <pre>
   *         f(x) = sgn( g(x) );
   *      d( f(x) ) = 0;
   * </pre>
   * Note: in general sgn'(x) = 2δ(x) where δ(x) is the Dirac delta function
   */
  public differentiate(_byVar: string): MathExpression {
    return new ConstantExpression(0)
  }
}
