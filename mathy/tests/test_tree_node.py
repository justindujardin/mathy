from ..core.tree import BinaryTreeNode

def test_tree_node_constructor():
    tree = BinaryTreeNode(BinaryTreeNode(), BinaryTreeNode())
    assert tree.left is not None and tree.right is not None
    count = 0

    def node_visit(node, depth, data):
        nonlocal count
        count = count + 1

    tree.visitInorder(node_visit)
    assert count == 3
    assert tree.left.parent == tree
    assert tree.right.parent == tree

