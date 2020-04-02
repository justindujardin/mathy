from mathy.core.tree import BinaryTreeNode, BinarySearchTree, STOP
import numpy as np
import pytest


def test_tree_node_constructor():
    """verify that the children passed in the constructor are properly assigned,
    and that their `parent` variables are also set to the root node properly."""

    tree = BinaryTreeNode(BinaryTreeNode(), BinaryTreeNode())
    assert tree.left is not None and tree.right is not None
    count = 0

    def node_visit(node, depth, data):
        nonlocal count
        count = count + 1

    tree.visit_inorder(node_visit)
    assert count == 3
    assert tree.left.parent == tree
    assert tree.right.parent == tree


def test_tree_node_clone():
    """check to be sure that when we clone off a known node of the tree
    that its children, and only its children, are still searchable
    from the cloned tree."""
    tree = BinarySearchTree(0)
    for i in range(26):
        tree.insert(i)
    fifteen = tree.find(15)
    assert fifteen is not None
    clone = fifteen.clone()
    # The clone node becomes the root of the new tree
    assert clone.get_root() == clone
    count = 0

    def node_visit(node, depth, data):
        nonlocal count
        count = count + 1

    clone.visit_inorder(node_visit)
    # clone node has expected 11 remaining nodes
    assert count == 11


def test_tree_node_is_leaf():

    """check that the known extremes of a tree are reported as leaf nodes
    and that all other known non - extremes are not. """
    tree = BinarySearchTree(0)
    for i in range(-1, 6):
        tree.insert(i)
    assert tree.find(-1).is_leaf() is True
    assert tree.find(5).is_leaf() is True
    for i in range(4):
        assert tree.find(i).is_leaf() is False


def test_tree_node_rotate():
    """test to ensure that rotations do not compromise the search tree
    by randomly rotating nodes and verifying that all known numbers can
    still be found. """
    tree = BinarySearchTree(0)
    values = []
    for i in range(-5, 6):
        values.append(i)
        tree.insert(i)
    for i in range(1000):
        index = np.random.randint(0, len(values))
        node = tree.find(values[index])
        node.rotate()
    for v in values:
        assert tree.find(v) is not None


def test_tree_node_visit_stop():
    """Verify that tree visits can be stopped by returning the STOP constant"""

    values = [-1, 0, 1]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)
    total = 0

    def visit_pre(node, depth, data):
        nonlocal total
        total += 1
        if node.key == -1:
            return STOP

    tree.visit_preorder(visit_pre)
    # preorder stops at second node
    assert total == 2

    total = 0

    def visit_in(node, depth, data):
        nonlocal total
        total += 1
        if node.key == -1:
            return STOP

    tree.visit_inorder(visit_in)
    # inorder stops at first node
    assert total == 1

    total = 0

    def visit_post(node, depth, data):
        nonlocal total
        total += 1
        if node.key == -1:
            return STOP

    tree.visit_postorder(visit_post)
    # postorder stops at first node
    assert total == 1


def test_tree_node_visit_preorder():
    values = [-1, 0, 1]
    order = [0, -1, 1]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)

    def node_visit(node, depth, data):
        nonlocal order
        assert node.key == order.pop(0)

    tree.visit_preorder(node_visit)


def test_tree_node_visit_inorder():
    values = [-1, 0, 1]
    order = [-1, 0, 1]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)

    def node_visit(node, depth, data):
        nonlocal order
        assert node.key == order.pop(0)

    tree.visit_inorder(node_visit)


def test_tree_node_visit_postorder():
    values = [-1, 0, 1]
    order = [-1, 1, 0]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)

    def node_visit(node, depth, data):
        nonlocal order
        assert node.key == order.pop(0)

    tree.visit_postorder(node_visit)


def test_tree_node_get_root():
    tree = BinarySearchTree(0)
    values = list(range(-5, 6))
    for i in values:
        tree.insert(i)
    for i in values:
        assert tree.find(i).get_root() == tree


def test_tree_node_set_left():
    one = BinaryTreeNode()
    two = BinaryTreeNode()
    three = BinaryTreeNode()
    one.set_left(two)
    assert one.left == two
    assert two.parent == one
    with pytest.raises(ValueError):
        # Cannot set self to child
        one.set_right(one)

    # can clear `.parent` on a child that is replaced
    assert two.parent == one
    one.set_left(three, clear_old_child_parent=True)
    assert two.parent is None


def test_tree_node_set_right():
    one = BinaryTreeNode()
    two = BinaryTreeNode()
    three = BinaryTreeNode()
    one.set_right(two)
    assert one.right == two
    assert two.parent == one

    with pytest.raises(ValueError):
        # Cannot set self to child
        one.set_right(one)

    # can clear `.parent` on a child that is replaced
    assert two.parent == one
    one.set_right(three, clear_old_child_parent=True)
    assert two.parent is None


def test_tree_node_get_side():
    values = list(range(-4, 0)) + list(range(1, 5))
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)
    node = tree.find(-4)
    assert node.parent.get_side(node) == "left"
    node = tree.find(4)
    assert node.parent.get_side(node) == "right"
    with pytest.raises(ValueError):
        # Raises an error if the child does not belong to this parent
        node.parent.get_side(BinaryTreeNode())


def test_tree_node_set_side():
    tree = BinaryTreeNode()
    one = BinaryTreeNode()
    two = BinaryTreeNode()
    tree.set_side(one, "left")
    assert tree.left == one
    tree.set_side(two, "right")
    assert tree.right == two
    with pytest.raises(ValueError):
        # error if side name is not known
        tree.set_side(one, "rihgt")


def test_tree_node_get_children():
    values = [-2, -1, -3, 0, 1, 2]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)
    neg = tree.find(-2).get_children()
    assert len(neg) == 2
    assert neg[0].key == -3
    assert neg[1].key == -1
    one = tree.find(1).get_children()
    assert len(one) == 1
    assert one[0].key == 2
    two = tree.find(2).get_children()
    assert len(two) == 0


def test_tree_node_get_sibling():
    tree = BinaryTreeNode(BinaryTreeNode(), BinaryTreeNode())
    assert tree.left.get_sibling() == tree.right
    assert tree.right.get_sibling() == tree.left
    assert tree.get_sibling() is None
