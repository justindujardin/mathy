// ## Constants

// Return this from a node visit function to abort a tree visit.
export const STOP = 'stop'
// The constant representing the left child side of a node.
export const LEFT = 'left'
// The constant representing the right child side of a node.
export const RIGHT = 'right'

/**
 * The binary tree node is the base node for all of our trees, and provides a
 * rich set of methods for constructing, inspecting, and modifying them.
 *
 * The node itself defines the structure of the binary tree, having left and right
 * children, and a parent.
 */
export class BinaryTreeNode {
  //  Allow specifying children in the constructor
  constructor(
    public left: BinaryTreeNode | null = null,
    public right: BinaryTreeNode | null = null,
    public parent: BinaryTreeNode | null = null
  ) {
    if (left) {
      this.setLeft(left)
    }
    if (right) {
      this.setRight(right)
    }
  }

  /** Create a clone of this tree */
  public clone() {
    const result = new (this.constructor as any)()
    if (this.left) {
      result.setLeft(this.left.clone())
    }
    if (this.right) {
      result.setRight(this.right.clone())
    }
    return result
  }

  /** Is this node a leaf?  A node is a leaf if it has no children. */
  public isLeaf() {
    return !this.left && !this.right
  }

  /** Serialize the node as a string */
  public toString() {
    return `${this.left} ${this.right}`
  }

  /** Human readable name for this node. */
  public getName() {
    return 'BinaryTreeNode'
  }

  /** Serialize the node as JSON */
  public toJSON() {
    return {
      name: this.getName(),
      children: this.getChildren().map(c => c.toJSON())
    }
  }

  /**
   * Rotate a node, changing the structure of the tree, without modifying
   * the order of the nodes in the tree.
   */
  public rotate() {
    const node = this
    const { parent } = this
    if (!node || !parent) {
      return
    }
    const grandParent = parent.parent
    if (node === parent.left) {
      parent.setLeft(node.right)
      node.right = parent
      parent.parent = node
    } else {
      parent.setRight(node.left)
      node.left = parent
      parent.parent = node
    }
    node.parent = grandParent
    if (!grandParent) {
      return
    }
    if (parent === grandParent.left) {
      return (grandParent.left = node)
    }

    return (grandParent.right = node)
  }

  // **Tree Traversal**
  //
  // Each visit method accepts a function that will be invoked for each node in the
  // tree.  The callback function is passed three arguments: the node being
  // visited, the current depth in the tree, and a user specified data parameter.
  //
  // *Traversals may be canceled by returning `STOP` from any visit function.*

  // Preorder : *Visit -> Left -> Right*
  public visitPreorder(visitFunction, depth = 0, data?) {
    if (visitFunction && visitFunction(this, depth, data) === STOP) {
      return STOP
    }
    if (this.left && this.left.visitPreorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
    if (this.right && this.right.visitPreorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
  }
  // Inorder : *Left -> Visit -> Right*
  public visitInorder(visitFunction, depth = 0, data?) {
    if (this.left && this.left.visitInorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
    if (visitFunction && visitFunction(this, depth, data) === STOP) {
      return STOP
    }
    if (this.right && this.right.visitInorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
  }
  // Postorder : *Left -> Right -> Visit*
  public visitPostorder(visitFunction, depth = 0, data?) {
    if (this.left && this.left.visitPostorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
    if (this.right && this.right.visitPostorder(visitFunction, depth + 1, data) === STOP) {
      return STOP
    }
    if (visitFunction && visitFunction(this, depth, data) === STOP) {
      return STOP
    }
  }

  // Return the root element of this tree
  public getRoot<T extends BinaryTreeNode>(): T {
    let result: BinaryTreeNode = this
    while (result.parent) {
      result = result.parent
    }
    return result as T
  }

  // **Child Management**
  //
  // Methods for setting the children on this expression.  These take care of
  // making sure that the proper parent assignments also take place.

  // Set the left node to the passed `child`
  public setLeft(child) {
    this.left = child
    if (this.left) {
      this.left.parent = this
    }
    return this
  }

  // Set the right node to the passed `child`
  public setRight(child) {
    this.right = child
    if (this.right) {
      this.right.parent = this
    }
    return this
  }

  // Determine whether the given `child` is the left or right child of this node
  public getSide(child) {
    if (child === this.left) {
      return LEFT
    }
    if (child === this.right) {
      return RIGHT
    }
    throw new Error('BinaryTreeNode.getSide: not a child of this node')
  }

  // Set a new `child` on the given `side`
  public setSide(child, side) {
    if (side === LEFT) {
      return this.setLeft(child)
    }
    if (side === RIGHT) {
      return this.setRight(child)
    }
    throw new Error('BinaryTreeNode.setSide: Invalid side')
  }

  // Get children as an array.  If there are two children, the first object will
  // always represent the left child, and the second will represent the right.
  public getChildren(): BinaryTreeNode[] {
    const result: BinaryTreeNode[] = []
    if (this.left) {
      result.push(this.left)
    }
    if (this.right) {
      result.push(this.right)
    }
    return result
  }

  // Get the sibling node of this node.  If there is no parent, or the node has no
  // sibling, the return value will be undefined.
  public getSibling(): BinaryTreeNode | null {
    if (!this.parent) {
      return null
    }
    if (this.parent.left === this) {
      return this.parent.right
    }
    if (this.parent.right === this) {
      return this.parent.left
    }
    return null
  }
}

/**
 * A very simple binary search tree that relies on keys that support
 * operator value comparison.
 * @class
 */
export class BinarySearchTree extends BinaryTreeNode {
  constructor(public key: string) {
    super()
  }
  public clone() {
    const result = super.clone()
    result.key = this.key
    return result
  }

  // Insert a node in the tree with the specified key.
  public insert(key) {
    let node = this.getRoot<BinarySearchTree>()
    while (node) {
      if (key > node.key) {
        if (!node.right) {
          node.setRight(new BinarySearchTree(key))
          break
        }
        node = node.right as BinarySearchTree
      } else if (key < node.key) {
        if (!node.left) {
          node.setLeft(new BinarySearchTree(key))
          break
        }
        node = node.left as BinarySearchTree
      } else {
        break
      }
    }
    return this
  }
  // Find a node in the tree by its key and return it.  Return null if the key
  // is not found in the tree.
  public find(key) {
    let node = this.getRoot<BinarySearchTree>()
    while (node) {
      if (key > node.key) {
        if (!node.right) {
          return null
        }
        node = node.right as BinarySearchTree
        continue
      }
      if (key < node.key) {
        if (!node.left) {
          return null
        }
        node = node.left as BinarySearchTree
        continue
      }
      if (key === node.key) {
        return node
      }
      return null
    }
    return null
  }
}

// ## <a id="BinaryTreeTidier"></a>BinaryTreeTidier

interface ITidierExtreme {
  left: ITidierExtreme | null
  right: ITidierExtreme | null
  thread: ITidierExtreme | null
  level: number
  offset: number
}
/**
 * Implement a Reingold-Tilford 'tidier' tree layout algorithm.
 * @class
 */
export class BinaryTreeTidier {
  // Assign x/y values to all nodes in the tree, and return an object containing
  // the measurements of the tree.
  public layout(node, unitMultiplier = 1) {
    this.measure(node)
    return this.transform(node, 0, unitMultiplier)
  }

  // Computer relative tree node positions
  public measure(node, level = 0, extremes: ITidierExtreme | null = null) {
    if (extremes == null) {
      extremes = { left: null, right: null, thread: null, level: 0, offset: 0 }
    }
    // left and right subtree extreme leaf nodes
    const leftExtremes: ITidierExtreme | null = {
      left: null,
      right: null,
      thread: null,
      level: 0,
      offset: 0
    }
    const rightExtremes: ITidierExtreme | null = {
      left: null,
      right: null,
      thread: null,
      level: 0,
      offset: 0
    }

    // separation at the root of the current subtree, as well as at the current level.
    let currentSeparation = 0
    let rootSeparation = 0
    const minimumSeparation = 1

    // The offset from left/right children to the root of the current subtree.
    let leftOffsetSum = 0
    let rightOffsetSum = 0

    // Avoid selecting as extreme
    if (!node) {
      if (extremes.left != null) {
        extremes.left.level = -1
      }
      if (extremes.right != null) {
        extremes.right.level = -1
      }
      return
    }

    // Assign the `node.y`, note the left/right child nodes, and recurse into the tree.
    node.y = level
    let { left } = node
    let { right } = node
    this.measure(left, level + 1, leftExtremes)
    this.measure(right, level + 1, rightExtremes)

    // A leaf is both the leftmost and rightmost node on the lowest level of the
    // subtree consisting of itself.
    if (!node.right && !node.left) {
      node.offset = 0
      extremes.right = extremes.left = node
      return extremes
    }

    // if only a single child, assign the next available offset and return.
    if (!node.right || !node.left) {
      node.offset = minimumSeparation
      extremes.right = extremes.left = node.left ? node.left : node.right
      return
    }

    // Set the current separation to the minimum separation for the root of the
    // subtree.
    currentSeparation = minimumSeparation
    leftOffsetSum = rightOffsetSum = 0

    // Traverse the subtrees until one of them is exhausted, pushing them apart
    // as needed.
    let loops = 0
    while (left && right) {
      loops++
      if (loops > 100000) {
        throw new Error('An impossibly large tree perhaps?')
      }
      if (currentSeparation < minimumSeparation) {
        rootSeparation += minimumSeparation - currentSeparation
        currentSeparation = minimumSeparation
      }

      if (left.right) {
        leftOffsetSum += left.offset
        currentSeparation -= left.offset
        left = left.thread || left.right
      } else {
        leftOffsetSum -= left.offset
        currentSeparation += left.offset
        left = left.thread || left.left
      }

      if (right.left) {
        rightOffsetSum -= right.offset
        currentSeparation -= right.offset
        right = right.thread || right.left
      } else {
        rightOffsetSum += right.offset
        currentSeparation += right.offset
        right = right.thread || right.right
      }
    }

    // Set the root offset, and include it in the accumulated offsets.
    node.offset = (rootSeparation + 1) / 2
    leftOffsetSum -= node.offset
    rightOffsetSum += node.offset

    // Update right and left extremes
    const rightLeftLevel = rightExtremes.left ? rightExtremes.left.level : -1
    const leftLeftLevel = leftExtremes.left ? leftExtremes.left.level : -1
    if (rightLeftLevel > leftLeftLevel || !node.left) {
      extremes.left = rightExtremes.left
      if (extremes.left) {
        extremes.left.offset += node.offset
      }
    } else {
      extremes.left = leftExtremes.left
      if (extremes.left) {
        extremes.left.offset -= node.offset
      }
    }
    const leftRightLevel = leftExtremes.right ? leftExtremes.right.level : -1
    const rightRightLevel = rightExtremes.right ? rightExtremes.right.level : -1
    if (leftRightLevel > rightRightLevel || !node.right) {
      extremes.right = leftExtremes.right
      if (extremes.right) {
        extremes.right.offset -= node.offset
      }
    } else {
      extremes.right = rightExtremes.right
      if (extremes.right) {
        extremes.right.offset += node.offset
      }
    }

    // If the subtrees have uneven heights, check to see if they need to be
    // threaded.  If threading is required, it will affect only one node.
    if (left && left !== node.left && rightExtremes && rightExtremes.right) {
      rightExtremes.right.thread = left
      rightExtremes.right.offset = Math.abs(rightExtremes.right.offset + node.offset - leftOffsetSum)
    } else if (right && right !== node.right && leftExtremes && leftExtremes.left) {
      leftExtremes.left.thread = right
      leftExtremes.left.offset = Math.abs(leftExtremes.left.offset - node.offset - rightOffsetSum)
    }

    // Return this
    return this
  }

  // Transform relative to absolute coordinates, and measure the bounds of the tree.
  // Return a measurement of the tree in output units.
  public transform(node, x = 0, unitMultiplier = 1, measure?) {
    if (!measure) {
      measure = { minX: 10000, maxX: 0, minY: 10000, maxY: 0 }
    }
    if (!node) {
      return measure
    }
    node.x = x * unitMultiplier
    node.y *= unitMultiplier
    this.transform(node.left, x - node.offset, unitMultiplier, measure)
    this.transform(node.right, x + node.offset, unitMultiplier, measure)
    if (measure.minY > node.y) {
      measure.minY = node.y
    }
    if (measure.maxY < node.y) {
      measure.maxY = node.y
    }
    if (measure.minX > node.x) {
      measure.minX = node.x
    }
    if (measure.maxX < node.x) {
      measure.maxX = node.x
    }
    measure.width = Math.abs(measure.minX - measure.maxX)
    measure.height = Math.abs(measure.minY - measure.maxY)
    measure.centerX = measure.minX + measure.width / 2
    measure.centerY = measure.minY + measure.height / 2
    return measure
  }
}
