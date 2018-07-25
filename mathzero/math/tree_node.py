import time

# ## Constants

# Return this from a node visit function to abort a tree visit.
STOP = "stop"
# The constant representing the left child side of a node.
LEFT = "left"
# The constant representing the right child side of a node.
RIGHT = "right"

class BinaryTreeNode:
    """
  The binary tree node is the base node for all of our trees, and provides a
  rich set of methods for constructing, inspecting, and modifying them.
  The node itself defines the structure of the binary tree, having left and right
  children, and a parent.
  """

    left: "BinaryTreeNode"
    right: "BinaryTreeNode"
    parent: "BinaryTreeNode"
    #  Allow specifying children in the constructor
    def __init__(self, left=None, right=None, parent=None):
        self.setLeft(left)
        self.setRight(right)
        self.parent = parent

    def clone(self):
        """Create a clone of this tree"""
        result = self.__class__()
        if self.left:
            result.setLeft(self.left.clone())

        if self.right:
            result.setRight(self.right.clone())
        return result

    def isLeaf(self):
        """Is this node a leaf?  A node is a leaf if it has no children."""
        return not self.left and not self.right

    def __str__(self):
        """Serialize the node as a str"""
        return "{} {}".format(self.left, self.right)

    def getName(self):
        """Human readable name for this node."""
        return "BinaryTreeNode"

    def toJSON(self):
        """Serialize the node as JSON"""
        return dotdict(
            {
                "name": self.getName(),
                "children": [c.toJSON() for c in self.getChildren()],
            }
        )

    def rotate(self):
        """
    Rotate a node, changing the structure of the tree, without modifying
    the order of the nodes in the tree.
    """
        node = self
        parent = self.parent
        if not node or not parent:
            return self

        grandParent = parent.parent
        if node == parent.left:
            parent.setLeft(node.right)
            node.right = parent
            parent.parent = node
        else:
            parent.setRight(node.left)
            node.left = parent
            parent.parent = node

        node.parent = grandParent
        if not grandParent:
            return self

        if parent == grandParent.left:
            grandParent.left = node
        else:
            grandParent.right = node
        return self

    # **Tree Traversal**
    #
    # Each visit method accepts a function that will be invoked for each node in the
    # tree.  The callback function is passed three arguments: the node being
    # visited, the current depth in the tree, and a user specified data parameter.
    #
    # *Traversals may be canceled by returning `STOP` from any visit function.*

    # Preorder : *Visit -> Left -> Right*
    def visitPreorder(self, visitFunction, depth=0, data=None):
        if visitFunction and visitFunction(self, depth, data) == STOP:
            return STOP

        if (
            self.left
            and self.left.visitPreorder(visitFunction, depth + 1, data) == STOP
        ):
            return STOP

        if (
            self.right
            and self.right.visitPreorder(visitFunction, depth + 1, data) == STOP
        ):
            return STOP

    # Inorder : *Left -> Visit -> Right*
    def visitInorder(self, visitFunction, depth=0, data=None):
        if self.left and self.left.visitInorder(visitFunction, depth + 1, data) == STOP:
            return STOP

        if visitFunction and visitFunction(self, depth, data) == STOP:
            return STOP

        if (
            self.right
            and self.right.visitInorder(visitFunction, depth + 1, data) == STOP
        ):
            return STOP

    # Postorder : *Left -> Right -> Visit*
    def visitPostorder(self, visitFunction, depth=0, data=None):
        if (
            self.left
            and self.left.visitPostorder(visitFunction, depth + 1, data) == STOP
        ):
            return STOP

        if (
            self.right
            and self.right.visitPostorder(visitFunction, depth + 1, data) == STOP
        ):
            return STOP

        if visitFunction and visitFunction(self, depth, data) == STOP:
            return STOP

    # Return the root element of this tree
    def getRoot(self) -> "BinaryTreeNode":
        result = self
        while result.parent:
            result = result.parent

        return result

    # **Child Management**
    #
    # Methods for setting the children on this expression.  These take care of
    # making sure that the proper parent assignments also take place.

    # Set the left node to the passed `child`
    def setLeft(self, child) -> "BinaryTreeNode":
        self.left = child
        if self.left:
            self.left.parent = self

        return self

    # Set the right node to the passed `child`
    def setRight(self, child) -> "BinaryTreeNode":
        self.right = child
        if self.right:
            self.right.parent = self

        return self

    # Determine whether the given `child` is the left or right child of this node
    def getSide(self, child):
        if child == self.left:
            return LEFT

        if child == self.right:
            return RIGHT

        raise Exception("BinaryTreeNode.getSide: not a child of this node")

    # Set a new `child` on the given `side`
    def setSide(self, child, side):
        if side == LEFT:
            return self.setLeft(child)

        if side == RIGHT:
            return self.setRight(child)

        raise Exception("BinaryTreeNode.setSide: Invalid side")

    # Get children as an array.  If there are two children, the first object will
    # always represent the left child, and the second will represent the right.
    def getChildren(self):
        result = []
        if self.left:
            result.append(self.left)

        if self.right:
            result.append(self.right)

        return result

    # Get the sibling node of this node.  If there is no parent, or the node has no
    # sibling, the return value will be None.
    def getSibling(self) -> "BinaryTreeNode":
        if not self.parent:
            return None

        if self.parent.left == self:
            return self.parent.right

        if self.parent.right == self:
            return self.parent.left

        return None


class BinarySearchTree(BinaryTreeNode):
    """A binary search tree by key"""

    key: str

    def __init__(self, key: str):
        super()
        self.key = key

    def clone(self) -> BinaryTreeNode:
        result = super().clone()
        result.key = self.key
        return result

    # Insert a node in the tree with the specified key.
    def insert(self, key) -> BinaryTreeNode:
        node = self.getRoot()
        while node:
            if key > node.key:
                if not node.right:
                    node.setRight(BinarySearchTree(key))
                    break
                node = node.right
            elif key < node.key:
                if not node.left:
                    node.setLeft(BinarySearchTree(key))
                    break

                node = node.left
            else:
                break
        return self

    # Find a node in the tree by its key and return it.  Return None if the key
    # is not found in the tree.
    def find(self, key) -> BinaryTreeNode:
        node = self.getRoot()
        while node:
            if key > node.key:
                if not node.right:
                    return None

                node = node.right
                continue

            if key < node.key:
                if not node.left:
                    return None

                node = node.left
                continue

            if key == node.key:
                return node

            return None

        return None


# ## <a id="BinaryTreeTidier"></a>BinaryTreeTidier


class TidierExtreme:
    left: "TidierExtreme"
    right: "TidierExtreme"
    thread: "TidierExtreme"
    level: int
    offset: int

    def __init__(self):
        self.left = None
        self.right = None
        self.thread = None
        self.left = 0
        self.offset = 0


class BinaryTreeTidier:
    """Implement a Reingold-Tilford 'tidier' tree layout algorithm."""

    # Assign x/y values to all nodes in the tree, and return an object containing
    # the measurements of the tree.
    def layout(self, node, unitMultiplier=1):
        self.measure(node)
        return self.transform(node, 0, unitMultiplier)

    # Computer relative tree node positions
    def measure(self, node, level=0, extremes: TidierExtreme = None):
        if extremes == None:
            extremes = TidierExtreme()

        # left and right subtree extreme leaf nodes
        leftExtremes = TidierExtreme()
        rightExtremes = TidierExtreme()

        # separation at the root of the current subtree, as well as at the current level.
        currentSeparation = 0
        rootSeparation = 0
        minimumSeparation = 1

        # The offset from left/right children to the root of the current subtree.
        leftOffsetSum = 0
        rightOffsetSum = 0

        # Avoid selecting as extreme
        if not node:
            if extremes.left != None:
                extremes.left.level = -1

            if extremes.right != None:
                extremes.right.level = -1

            return

        # Assign the `node.y`, note the left/right child nodes, and recurse into the tree.
        node.y = level
        left = node.left
        right = node.right
        self.measure(left, level + 1, leftExtremes)
        self.measure(right, level + 1, rightExtremes)

        # A leaf is both the leftmost and rightmost node on the lowest level of the
        # subtree consisting of itself.
        if not node.right and not node.left:
            node.offset = 0
            extremes.right = extremes.left = node
            return extremes

        # if only a single child, assign the next available offset and return.
        if not node.right or not node.left:
            node.offset = minimumSeparation
            extremes.right = extremes.left = node.left if node.left else node.right
            return

        # Set the current separation to the minimum separation for the root of the
        # subtree.
        currentSeparation = minimumSeparation
        leftOffsetSum = rightOffsetSum = 0

        # Traverse the subtrees until one of them is exhausted, pushing them apart
        # as needed.
        loops = 0
        while left and right:
            loops = loops + 1
            if loops > 100000:
                raise Exception("An impossibly large tree perhaps?")

            if currentSeparation < minimumSeparation:
                rootSeparation += minimumSeparation - currentSeparation
                currentSeparation = minimumSeparation

            if left.right:
                leftOffsetSum += left.offset
                currentSeparation -= left.offset
                left = left.thread or left.right
            else:
                leftOffsetSum -= left.offset
                currentSeparation += left.offset
                left = left.thread or left.left

            if right.left:
                rightOffsetSum -= right.offset
                currentSeparation -= right.offset
                right = right.thread or right.left
            else:
                rightOffsetSum += right.offset
                currentSeparation += right.offset
                right = right.thread or right.right

        # Set the root offset, and include it in the accumulated offsets.
        node.offset = (rootSeparation + 1) / 2
        leftOffsetSum -= node.offset
        rightOffsetSum += node.offset

        # Update right and left extremes
        rightLeftLevel = rightExtremes.left.level if rightExtremes.left else -1
        leftLeftLevel = leftExtremes.left.level if leftExtremes.left else -1
        if rightLeftLevel > leftLeftLevel or not node.left:
            extremes.left = rightExtremes.left
            if extremes.left:
                extremes.left.offset += node.offset

        else:
            extremes.left = leftExtremes.left
            if extremes.left:
                extremes.left.offset -= node.offset

        leftRightLevel = leftExtremes.right.level if leftExtremes.right else -1
        rightRightLevel = rightExtremes.right.level if rightExtremes.right else -1
        if leftRightLevel > rightRightLevel or not node.right:
            extremes.right = leftExtremes.right
            if extremes.right:
                extremes.right.offset -= node.offset

        else:
            extremes.right = rightExtremes.right
            if extremes.right:
                extremes.right.offset += node.offset

        # If the subtrees have uneven heights, check to see if they need to be
        # threaded.  If threading is required, it will affect only one node.
        if left and left != node.left and rightExtremes and rightExtremes.right:
            rightExtremes.right.thread = left
            rightExtremes.right.offset = Math.abs(
                rightExtremes.right.offset + node.offset - leftOffsetSum
            )
        elif right and right != node.right and leftExtremes and leftExtremes.left:
            leftExtremes.left.thread = right
            leftExtremes.left.offset = Math.abs(
                leftExtremes.left.offset - node.offset - rightOffsetSum
            )

        # Return self
        return self

    # Transform relative to absolute coordinates, and measure the bounds of the tree.
    # Return a measurement of the tree in output units.
    def transform(self, node, x=0, unitMultiplier=1, measure=None):
        if measure is None:
            measure = dotdict({"minX": 10000, "maxX": 0, "minY": 10000, "maxY": 0})
        if not node:
            return measure

        node.x = x * unitMultiplier
        node.y *= unitMultiplier
        self.transform(node.left, x - node.offset, unitMultiplier, measure)
        self.transform(node.right, x + node.offset, unitMultiplier, measure)
        if measure.minY > node.y:
            measure.minY = node.y

        if measure.maxY < node.y:
            measure.maxY = node.y

        if measure.minX > node.x:
            measure.minX = node.x

        if measure.maxX < node.x:
            measure.maxX = node.x

        measure.width = Math.abs(measure.minX - measure.maxX)
        measure.height = Math.abs(measure.minY - measure.maxY)
        measure.centerX = measure.minX + measure.width / 2
        measure.centerY = measure.minY + measure.height / 2
        return measure


if __name__ == "__main__":
    print("cool")

