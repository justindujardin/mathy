from mathy.core.tree import BinaryTreeNode, BinarySearchTree, STOP
from mathy.core.layout import TreeLayout, TreeMeasurement
import numpy as np
import pytest


def test_layout_tidier():
    tree = BinarySearchTree(0)
    for i in range(-10, 10):
        tree.insert(i)
    m: TreeMeasurement = TreeLayout().layout(tree)
    assert m is not None


def test_layout_tidier_aesthetic_one():
    """Verify layouts satisfy the "y-coordinate sharing" aesthetic.

    Build a tree, gather up all the nodes, group them by depth, and
    assert that each node in a depth group has the same y - coordinate"""
    tree = BinarySearchTree(0)
    for i in range(-5, 5):
        tree.insert(i)
    m: TreeMeasurement = TreeLayout().layout(tree)
    assert m is not None

    groups = {}

    def node_visit(node, depth, data):
        nonlocal groups
        if depth not in groups:
            groups[depth] = []
        groups[depth].append(node)

    tree.visit_preorder(node_visit)
    for k, v in groups.items():
        y_coords = [n.y for n in v]
        # converting to a set removes duplicates
        assert len(set(y_coords)) == 1


def test_layout_tidier_aesthetic_two():
    """Verify layouts satisfy the "children relative position" aesthetic.

    Build a tree, visit each node and assert that the left/right children are
    positioned to the left and right of its own coordinate.
    """
    tree = BinarySearchTree(0)
    for i in [-2, -1, -3, 2, 3, 1]:
        tree.insert(i)
    # Discard the measurement because each node has x/y attached
    TreeLayout().layout(tree)

    def node_visit(node: BinaryTreeNode, depth, data):
        if node.parent is None:
            return
        side = node.parent.get_side(node)
        if side == "left":
            # left child should be to the left of the parent (-x)
            assert node.x < node.parent.x
        elif side == "right":
            # left child should be to the left of the parent (+x)
            assert node.x > node.parent.x

    tree.visit_postorder(node_visit)


def test_layout_tidier_aesthetic_three():
    """Verify layouts satisfy the "equidistant children" aesthetic.

    Build a tree, visit each node and assert that its children are
    equidistant from it."""
    values = [-2, -1, -3, 2, 3, 1]
    tree = BinarySearchTree(0)
    for i in values:
        tree.insert(i)
    TreeLayout().layout(tree)

    def node_visit(node: BinaryTreeNode, depth: int, data):
        children = node.get_children()
        if len(children) != 2:
            return
        left = children[0].x
        right = children[1].x

        expected = left + abs(left - right) / 2
        # Parent node position matches expectation
        assert expected == node.x

    tree.visit_postorder(node_visit)


def test_layout_tidier_aesthetic_four_reflections():
    """Verify layouts satisfy the "subtree reflections" aesthetic.

    Build a tree with subtrees that are reflections of one another, and
    visit each node asserting that its mirror node is on the opposite
    side of the tree with the same distance to the root.
    """
    tree = BinarySearchTree(0)
    values = [-2, -1, -4, -3, -5, 2, 1, 4, 3, 5]
    for i in values:
        tree.insert(i)
    TreeLayout().layout(tree)
    for i in range(6):
        left = tree.find(-i)
        right = tree.find(i)
        left_dist = abs(left.x - tree.x)
        right_dist = abs(right.x - tree.x)
        assert left_dist == right_dist


def test_layout_tidier_aesthetic_four_identical_subtrees():
    """Verify layouts satisfy the "subtree identical regardless of position" aesthetic.

    Build a tree with known identical subtrees that are in different positions of the
    tree, and verify that their children have the same relative positions to them.
    """
    tree = BinarySearchTree(0)
    values = [7, 4, 3, 5, 13, 12, 14, -3, -6, -2, -7, -13, -14, -12]
    for i in values:
        tree.insert(i)
    TreeLayout().layout(tree)

    first_left_distance = None
    first_right_distance = None

    def node_visit(node, depth: int, data):
        nonlocal first_left_distance, first_right_distance
        # We only want to assert about the specific subtrees +-(13, 14, 12)
        if node.key != 13 and node.key != -13:
            return
        # expect lesser value to the left
        assert node.left.key == node.key - 1
        # expect greater value to the left
        assert node.right.key == node.key + 1

        less = node.left.x
        more = node.right.x
        left_dist = less - node.x
        right_dist = more - node.x
        if first_right_distance and first_left_distance:
            assert left_dist == first_left_distance
            assert right_dist == first_right_distance
        else:
            first_left_distance = left_dist
            first_right_distance = right_dist

    tree.visit_postorder(node_visit)
