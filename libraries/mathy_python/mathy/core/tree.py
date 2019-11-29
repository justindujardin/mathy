import time
import curses
import uuid
from typing import Optional, Union

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

    _idCounter = 0

    # Tree layout mutations. Thanks, 2009 Justin. :( :(
    x: Optional[float]
    y: Optional[float]
    offset: Optional[float]

    #  Allow specifying children in the constructor
    def __init__(self, left=None, right=None, parent=None, id=None):
        if id is None:
            BinaryTreeNode._idCounter = BinaryTreeNode._idCounter + 1
            id = "mn-{}".format(BinaryTreeNode._idCounter)
        self.id = id
        self.left = None
        self.right = None
        self.set_left(left)
        self.set_right(right)
        self.parent = parent

    def clone(self):
        """Create a clone of this tree"""
        result = self.__class__()
        result.id = self.id
        if self.left:
            result.set_left(self.left.clone())

        if self.right:
            result.set_right(self.right.clone())
        return result

    def is_leaf(self):
        """Is this node a leaf?  A node is a leaf if it has no children."""
        return not self.left and not self.right

    def __str__(self):
        """Serialize the node as a str"""
        return "{} {}".format(self.left, self.right)

    @property
    def name(self):
        """Human readable name for this node."""
        return "BinaryTreeNode"

    def to_json(self):
        """Serialize the node as JSON"""
        return {
            "name": self.name,
            "children": [c.to_json() for c in self.get_children()],
        }

    def rotate(self):
        """
        Rotate a node, changing the structure of the tree, without modifying
        the order of the nodes in the tree.
        """
        node = self
        parent = self.parent
        if not node or not parent:
            return self

        grand_parent = parent.parent
        if node == parent.left:
            parent.set_left(node.right)
            node.right = parent
            parent.parent = node
        else:
            parent.set_right(node.left)
            node.left = parent
            parent.parent = node

        node.parent = grand_parent
        if not grand_parent:
            return self

        if parent == grand_parent.left:
            grand_parent.left = node
        else:
            grand_parent.right = node
        return self

    # **Tree Traversal**
    #
    # Each visit method accepts a function that will be invoked for each node in the
    # tree.  The callback function is passed three arguments: the node being
    # visited, the current depth in the tree, and a user specified data parameter.
    #
    # *Traversals may be canceled by returning `STOP` from any visit function.*

    # Preorder : *Visit -> Left -> Right*
    def visit_preorder(self, visit_fn, depth=0, data=None):
        if visit_fn and visit_fn(self, depth, data) == STOP:
            return STOP

        if self.left and self.left.visit_preorder(visit_fn, depth + 1, data) == STOP:
            return STOP

        if self.right and self.right.visit_preorder(visit_fn, depth + 1, data) == STOP:
            return STOP

    # Inorder : *Left -> Visit -> Right*
    def visit_inorder(self, visit_fn, depth=0, data=None):
        if self.left and self.left.visit_inorder(visit_fn, depth + 1, data) == STOP:
            return STOP

        if visit_fn and visit_fn(self, depth, data) == STOP:
            return STOP

        if self.right and self.right.visit_inorder(visit_fn, depth + 1, data) == STOP:
            return STOP

    # Postorder : *Left -> Right -> Visit*
    def visit_postorder(self, visit_fn, depth=0, data=None):
        if self.left and self.left.visit_postorder(visit_fn, depth + 1, data) == STOP:
            return STOP

        if self.right and self.right.visit_postorder(visit_fn, depth + 1, data) == STOP:
            return STOP

        if visit_fn and visit_fn(self, depth, data) == STOP:
            return STOP

    # Return the root element of this tree
    def get_root(self):
        result = self
        while result.parent:
            result = result.parent

        return result

    # **Child Management**
    #
    # Methods for setting the children on this expression.  These take care of
    # making sure that the proper parent assignments also take place.

    # Set the left node to the passed `child`
    def set_left(
        self, child: "BinaryTreeNode", clear_old_child_parent=False
    ) -> "BinaryTreeNode":
        if child == self:
            raise ValueError("nodes cannot be their own children")
        if self.left is not None and clear_old_child_parent:
            self.left.parent = None
        self.left = child
        if self.left:
            self.left.parent = self

        return self

    # Set the right node to the passed `child`
    def set_right(
        self, child: "BinaryTreeNode", clear_old_child_parent=False
    ) -> "BinaryTreeNode":
        if child == self:
            raise ValueError("nodes cannot be their own children")
        if self.right is not None and clear_old_child_parent:
            self.right.parent = None
        self.right = child
        if self.right:
            self.right.parent = self

        return self

    # Determine whether the given `child` is the left or right child of this node
    def get_side(self, child):
        if child == self.left:
            return LEFT

        if child == self.right:
            return RIGHT

        raise ValueError("BinaryTreeNode.get_side: not a child of this node")

    # Set a new `child` on the given `side`
    def set_side(self, child, side):
        if side == LEFT:
            return self.set_left(child)

        if side == RIGHT:
            return self.set_right(child)

        raise ValueError("BinaryTreeNode.set_side: Invalid side")

    # Get children as an array.  If there are two children, the first object will
    # always represent the left child, and the second will represent the right.
    def get_children(self):
        result = []
        if self.left:
            result.append(self.left)

        if self.right:
            result.append(self.right)

        return result

    # Get the sibling node of this node.  If there is no parent, or the node has no
    # sibling, the return value will be None.
    def get_sibling(self) -> Optional["BinaryTreeNode"]:
        if not self.parent:
            return None

        if self.parent.left == self:
            return self.parent.right

        if self.parent.right == self:
            return self.parent.left

        return None


class BinarySearchTree(BinaryTreeNode):
    """A binary search tree by key"""

    def __init__(self, key: Union[str, int, float] = None, **kwargs):
        super(BinarySearchTree, self).__init__(**kwargs)
        self.key = key

    def clone(self) -> BinaryTreeNode:
        result = super(BinarySearchTree, self).clone()
        result.key = self.key
        return result

    # Insert a node in the tree with the specified key.
    def insert(self, key) -> BinaryTreeNode:
        node = self.get_root()
        while node:
            if key > node.key:
                if not node.right:
                    node.set_right(BinarySearchTree(key))
                    break
                node = node.right
            elif key < node.key:
                if not node.left:
                    node.set_left(BinarySearchTree(key))
                    break

                node = node.left
            else:
                break
        return self

    # Find a node in the tree by its key and return it.  Return None if the key
    # is not found in the tree.
    def find(self, key) -> Optional[BinaryTreeNode]:
        node = self.get_root()
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
