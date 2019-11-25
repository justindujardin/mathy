import {
  AddExpression,
  SubtractExpression,
  ConstantExpression,
  VariableExpression,
  MultiplyExpression,
  PowerExpression,
  BinaryExpression,
  NegateExpression,
  MathExpression
} from './expressions'

import _ from 'lodash'
import { Variable } from '@tensorflow/tfjs'

/**
 * Generate a UUIDv4
 *
 * See: https://stackoverflow.com/questions/105034/create-guid-uuid-in-javascript
 */
export function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    // tslint:disable-next-line:no-bitwise
    const r = (Math.random() * 16) | 0
    // tslint:disable-next-line:no-bitwise
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/* istanbul ignore next */
/**
 * This exploits Typescript's control flow analysis to do exhaustive pattern
 * matching on switch statements that have tagged union types being switched
 * on.
 *
 * This is useful for prevent access errors and ensuring all cases are considered
 * when adding new features.
 */
export function exhaustiveCheck(_action: never) {
  // MAGIC!
}
/**
 * Unlink an expression from it's parent.
 *
 * 1. Clear expression references in `parent`
 * 2. Clear `parent` in expression
 * @param node The node to unlink
 */
export function unlink(node: MathExpression | null): MathExpression | null {
  if (!node) {
    return null
  }
  if (node.parent) {
    if (node === node.parent.left) {
      node.parent.setLeft(null)
    }
    if (node === node.parent.right) {
      node.parent.setRight(null)
    }
  }
  node.parent = null
  return node
}

export interface NumberFactors {
  [numberValue: string]: number
}

// Build a dictionary of factors for a given value that
// contains both arrangements of terms so that all factors are
// accessible by key.  That is, factoring 2 would return
//      result =
//        1 : 2
//        2 : 1
export function factor(value: number): NumberFactors {
  if (value === 0) {
    return {}
  }
  const sqrt = Math.sqrt(value) + 1
  const factors: { [numberValue: string]: number } = { 1: value }
  factors[value] = 1
  for (let i = 2, end = sqrt, asc = 2 <= end; asc ? i <= end : i >= end; asc ? i++ : i--) {
    if (value % i === 0) {
      const one = i
      const two = value / i
      factors[one] = two
      factors[two] = one
    }
  }
  return factors
}

export function isAddSubtract(node: MathExpression) {
  return node instanceof AddExpression || node instanceof SubtractExpression
}
export interface AddTermFactors {
  best: number
  left: number
  right: number
  all_left: NumberFactors
  all_right: NumberFactors
  variable: string | null
  exponent: number | null
  leftExponent: number | null
  rightExponent: number | null
  leftVariable: string | null
  rightVariable: string | null
}

export function factorAddTerms(node: MathExpression): AddTermFactors | boolean {
  if (!isAddSubtract(node) || !node.left || !node.right) {
    throw new Error('Can only factor add/substract nodes with valid left/rigth children')
  }
  const lTerm = getTerm(node.left)
  const rTerm = getTerm(node.right)
  if (!lTerm || !rTerm) {
    throw new Error(`Complex or unidentifiable term/s in ${node}`)
  }

  // Complex terms with multiple coefficients, simply first.
  if (lTerm.coefficients && lTerm.coefficients.length > 1) {
    return false
  }
  if (rTerm.coefficients && rTerm.coefficients.length > 1) {
    return false
  }

  // Common coefficients
  const lCoefficients = factor(lTerm.coefficients[0] || 1)
  const rCoefficients = factor(rTerm.coefficients[0] || 1)
  const common: number[] = []
  for (const k in rCoefficients) {
    if (k in lCoefficients) {
      common.push(parseInt(k, 10))
    }
  }
  const best = Math.max.apply(null, common)
  const result: Partial<AddTermFactors> = {
    best,
    left: lCoefficients[best],
    right: rCoefficients[best],
    all_left: lCoefficients,
    all_right: rCoefficients
  }

  // Common variables and powers
  const commonExp = lTerm.exponent && rTerm.exponent && lTerm.exponent === rTerm.exponent
  const expMatch = (lTerm.exponent || rTerm.exponent) && !commonExp ? false : true
  if (
    lTerm.variables[0] &&
    rTerm.variables[0] &&
    lTerm.variables[0] === rTerm.variables[0] &&
    expMatch
  ) {
    result.variable = lTerm.variables[0]
    result.exponent = lTerm.exponent
  }
  if (lTerm.exponent && lTerm.exponent !== result.exponent) {
    result.leftExponent = lTerm.exponent
  }
  if (rTerm.exponent && rTerm.exponent !== result.exponent) {
    result.rightExponent = rTerm.exponent
  }
  if (lTerm.variables[0] && lTerm.variables[0] !== result.variable) {
    result.leftVariable = lTerm.variables[0]
  }
  if (rTerm.variables[0] && rTerm.variables[0] !== result.variable) {
    result.rightVariable = rTerm.variables[0]
  }

  // cast to convince the compiler we've filled out the partial type above
  return result as AddTermFactors
}

/**
 * Create a new term node hierarchy from a given set of
 * term parameters. This takes into account removing implicit
 * coefficients of 1 where possible.
 * @param coefficient The number coefficient for the term
 * @param variable The variable (or undefined)
 * @param exponent The exponent to attach (or undefined)
 */
export function makeTerm(
  coefficient: number,
  variable?: string,
  exponent?: number
): MathExpression {
  const constExp = new ConstantExpression(coefficient)
  if (variable === undefined) {
    // const: (4, 13, ...)
    if (!exponent) {
      return constExp
    }
    // const^power: (4^2, 3^3, ...)
    return new PowerExpression(constExp, new ConstantExpression(exponent))
  }
  const varExp = new VariableExpression(variable)
  // 1*var: (1x, 1z) return implicit form that leaves off 1
  if (coefficient === 1 && !exponent) {
    return varExp
  }
  const multExp = new MultiplyExpression(constExp, varExp)
  if (!exponent) {
    return multExp
  }
  const expConstExp = new ConstantExpression(exponent)
  if (coefficient === 1) {
    return new PowerExpression(varExp, expConstExp)
  }
  return new PowerExpression(multExp, expConstExp)
}

export interface TermResult {
  coefficients: number[]
  node_coefficients: ConstantExpression[]
  exponent?: number
  variables: string[]
  node_variables: VariableExpression[]
  node_exponent?: PowerExpression
}

// Extract term information from the given node
//
export function getTerm(node: MathExpression): TermResult | false {
  let exponent
  const result: TermResult = {
    coefficients: [],
    exponent: undefined,
    node_coefficients: [],
    node_variables: [],
    variables: []
  }
  // Constant with add/sub parent should be OKAY.
  if (node instanceof ConstantExpression) {
    if (!node.parent || (node.parent && isAddSubtract(node.parent))) {
      result.coefficients = [node.value]
      result.node_coefficients = [node]
      return result
    }
  }
  // Variable with add/sub parent should be OKAY.
  if (node instanceof VariableExpression) {
    if (!node.parent || (node.parent && isAddSubtract(node.parent))) {
      result.variables = [node.identifier]
      result.node_variables = [node]
      return result
    }
  }

  // TODO: Comment resolution on whether +- is OKAY, and if not, why it breaks down.
  if (!isAddSubtract(node)) {
    if (
      node.findByType(AddExpression).length > 0 ||
      node.findByType(SubtractExpression).length > 0
    ) {
      return false
    }
  }

  // If another add is found on the left side of this node, and the right node
  // is _NOT_ a leaf, we cannot extract a term.  If it is a leaf, the term should be
  // just the right node.
  if (node.left && node.left.findByType(AddExpression).length > 0) {
    if (node.right && !node.right.isLeaf()) {
      return false
    }
  }
  if (node.right && node.right.findByType(AddExpression).length > 0) {
    return false
  }

  const exponents = node.findByType<PowerExpression>(PowerExpression)
  if (exponents.length > 0) {
    // Supports only single exponents in terms
    if (exponents.length !== 1) {
      return false
    }
    exponent = exponents[0]
    if (!(exponent.right instanceof ConstantExpression)) {
      throw new Error('getTerm supports constant term powers')
    }
    result.exponent = exponent.right.value
    result.node_exponent = exponent
  }

  const variables = node.findByType<VariableExpression>(VariableExpression)
  if (variables.length > 0) {
    result.variables = variables.map(v => v.identifier)
    result.node_variables = variables
  }

  let coefficients = node.findByType<ConstantExpression>(ConstantExpression)
  coefficients = _.reject(coefficients, function(n) {
    if (!n.parent || n.parent === node.parent) {
      return false
    }
    if (n.parent instanceof BinaryExpression && !(n.parent instanceof MultiplyExpression)) {
      return true
    }
    return false
  })
  if (coefficients.length > 0) {
    result.coefficients = _.map(coefficients, function(c) {
      let { value } = c
      if (c.parent instanceof NegateExpression) {
        value *= -1
      }
      return value
    })
    result.node_coefficients = coefficients
  }

  const empty =
    result.variables.length === 0 &&
    result.coefficients.length === 0 &&
    result.exponent === undefined
  if (empty) {
    return false
  }

  // consistently return an empty coefficients/variables array in case none exist.
  // this ensures that you can always reference coefficients[0] or variables[0] and
  // check that for truthiness, rather than having to check that the object property
  // `coefficients` or `variables` is not undefined and also the truthiness of index 0.
  if (!result.coefficients) {
    result.coefficients = []
  }
  if (!result.variables) {
    result.variables = []
  }
  return result
}

export function getTerms(node: MathExpression) {
  const terms = []
  let current: MathExpression | null = node
  while (current) {
    if (!isAddSubtract(current)) {
      terms.push(current)
      break
    } else {
      if (current.right) {
        terms.push(current.right)
      }
    }
    current = current.left
  }
  return terms
}
/**
 * @param {Object|MathExpression} one The first term {@link #getTerm}
 * @param {Object|MathExpression} two The second term {@link #getTerm}
 * @returns {Boolean} Whether the terms are like or not.
 */
export function termsAreLike(one: MathExpression, two: MathExpression) {
  // Both must be valid terms
  if (!one && !two) {
    return false
  }

  // Extract term info from each
  const oneTerm = getTerm(one)
  const twoTerm = getTerm(two)
  // If they're both falsy, they're not different
  if (!oneTerm && !twoTerm) {
    return false
  }
  // If one is falsy, they are different
  if (!oneTerm || !twoTerm) {
    return true
  }
  // Both must have variables, and they must match exactly.
  if (!oneTerm.variables || !twoTerm.variables) {
    return false
  }
  if (!oneTerm.variables.length || !twoTerm.variables.length) {
    return false
  }
  if (oneTerm.variables.length !== twoTerm.variables.length) {
    return false
  }
  let match = true
  _.each(oneTerm.variables, function(v, index) {
    if (twoTerm.variables[index] !== v) {
      return (match = false)
    }
  })
  if (!match) {
    return false
  }

  // Also, the exponents must match
  if (oneTerm.exponent !== twoTerm.exponent) {
    return false
  }

  // Same variables, and exponents.  Good to go.
  return true
}

// Negate helper

// `l - r = l + (-r)`
//
//             -                  +
//            / \                / \
//           /   \     ->       /   \
//          /     \            /     \
//         *       2          *       -
//        / \                / \       \
//       4   x              4   x       2
export function negate(node: MathExpression) {
  if (!node.left || !node.right) {
    throw new Error('cannot negate an expression with invalid left or right children')
  }
  const save = node.parent
  const saveSide = save != null ? save.getSide(node) : undefined
  unlink(node)
  const newNode = new AddExpression(node.left, new NegateExpression(node.right))
  if (save != null) {
    save.setSide(newNode, saveSide)
  }
  return newNode
}

// Determine if an expression represents a constant term
export function isConstTerm(node: MathExpression) {
  if (node instanceof ConstantExpression) {
    return true
  }
  if (node instanceof NegateExpression && isConstTerm(node.child)) {
    return true
  }
  return false
}

export function getTermConst(node: MathExpression) {
  if (node instanceof ConstantExpression) {
    return node.value
  }
  if (node.left instanceof ConstantExpression) {
    return node.left.value
  }
  if (node.right instanceof ConstantExpression) {
    return node.right.value
  }
  throw new Error('Unable to determine coefficient for expression')
}
