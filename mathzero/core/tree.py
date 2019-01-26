import time
import curses
import uuid

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

    #  Allow specifying children in the constructor
    def __init__(self, left=None, right=None, parent=None, id=None):
        if id is None:
            id = uuid.uuid4().hex
        self.id = id
        self.setLeft(left)
        self.setRight(right)
        self.parent = parent

    def clone(self):
        """Create a clone of this tree"""
        result = self.__class__()
        result.id = self.id
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

    @property
    def name(self):
        """Human readable name for this node."""
        return "BinaryTreeNode"

    def toJSON(self):
        """Serialize the node as JSON"""
        return {"name": self.name, "children": [c.toJSON() for c in self.getChildren()]}

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
