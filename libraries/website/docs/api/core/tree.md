# mathy.core.tree

## BinaryTreeNode
```python
BinaryTreeNode(self, left=None, right=None, parent=None, id=None)
```

The binary tree node is the base node for all of our trees, and provides a
rich set of methods for constructing, inspecting, and modifying them.
The node itself defines the structure of the binary tree, having left and right
children, and a parent.

### name
Human readable name for this node.
### clone
```python
BinaryTreeNode.clone(self)
```
Create a clone of this tree
### is_leaf
```python
BinaryTreeNode.is_leaf(self)
```
Is this node a leaf?  A node is a leaf if it has no children.
### to_json
```python
BinaryTreeNode.to_json(self)
```
Serialize the node as JSON
### rotate
```python
BinaryTreeNode.rotate(self)
```

Rotate a node, changing the structure of the tree, without modifying
the order of the nodes in the tree.

## BinarySearchTree
```python
BinarySearchTree(self, key:Union[str, int, float]=None, **kwargs)
```
A binary search tree by key
