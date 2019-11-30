# mathy.core.layout
Tree Layout
---

In order to help visualize, understand, and debug math trees and transformations to
them, Mathy implements a
[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=_blank} layout
algorithm that works with expression trees. It produces beautiful trees like:

`mathy:(2x^3 + y)(14 + 2.3y)`


## TreeLayout
```python
TreeLayout(self, /, *args, **kwargs)
```
Calculate a visual layout for input trees.
### layout
```python
TreeLayout.layout(self, node:mathy.core.tree.BinaryTreeNode, unit_x_multiplier=1, unit_y_multiplier=1) -> 'TreeMeasurement'
```
Assign x/y values to all nodes in the tree, and return an object containing
the measurements of the tree.

Returns a TreeMeasurement object that describes the bounds of the tree
### transform
```python
TreeLayout.transform(self, node:mathy.core.tree.BinaryTreeNode, x=0, unit_x_multiplier=1, unit_y_multiplier=1, measure=None) -> 'TreeMeasurement'
```
Transform relative to absolute coordinates, and measure the bounds of the tree.

Return a measurement of the tree in output units.
## TreeMeasurement
```python
TreeMeasurement(self)
```
Summary of the rendered tree
