# mathy.core.layout
Tree Layout
---

In order to help visualize, understand, and debug math trees and transformations to
them, Mathy implements a
[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=\_blank} layout
algorithm that works with expression trees. It produces beautiful trees like:

`mathy:(2x^3 + y)(14 + 2.3y)`


## TidierExtreme
```python
TidierExtreme(self)
```
Hold state about an extreme end of the tree during layout.
## TreeMeasurement
```python
TreeMeasurement(self)
```
Summary of the rendered tree
## TreeLayout
```python
TreeLayout(self, /, *args, **kwargs)
```
[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=\_blank}
`tidier` tree layout algorithm.

The tidier tree algorithm produces aesthetically pleasing compact trees with no
overlapping nodes regardless of the depth or complexity of the tree.
