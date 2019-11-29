"""Tree Layout
---

In order to help visualize, understand, and debug math trees and transformations to
them, Mathy implements a
[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=\_blank} layout
algorithm that works with expression trees. It produces beautiful trees like:

`mathy:(2x^3 + y)(14 + 2.3y)`

"""
from .tree import BinaryTreeNode


class TidierExtreme:
    """Hold state about an extreme end of the tree during layout."""

    def __init__(self):
        self.left = None
        self.right = None
        self.thread = None
        self.offset = 0


class TreeMeasurement:
    """Summary of the rendered tree"""

    def __init__(self):
        self.minX = 10000
        self.maxX = 0
        self.minY = 10000
        self.maxY = 0
        self.width = 0
        self.height = 0
        self.centerX = 0
        self.centerY = 0

    def __repr__(self):
        return """tree measurement: bounds({}, {}), min({}, {}), max({}, {}), center({}, {})""".format(
            self.width,
            self.height,
            self.minX,
            self.minY,
            self.maxX,
            self.maxY,
            self.centerX,
            self.centerY,
        )


class TreeLayout:
    """[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=\_blank}
     `tidier` tree layout algorithm.

    The tidier tree algorithm produces aesthetically pleasing compact trees with no
    overlapping nodes regardless of the depth or complexity of the tree."""

    # Assign x/y values to all nodes in the tree, and return an object containing
    # the measurements of the tree.
    def layout(self, node: BinaryTreeNode, unitMultiplierX=1, unitMultiplierY=1):
        self.measure(node)
        return self.transform(node, 0, unitMultiplierX, unitMultiplierY)

    # Compute relative tree node positions
    def measure(
        self, node: BinaryTreeNode, level=0, extremes: TidierExtreme = None
    ) -> TreeMeasurement:
        if extremes is None:
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
            if extremes.left is not None:
                extremes.left.level = -1

            if extremes.right is not None:
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
                left = getattr(left, "thread", left.right)
            else:
                leftOffsetSum -= left.offset
                currentSeparation += left.offset
                left = getattr(left, "thread", left.left)

            if right.left:
                rightOffsetSum -= right.offset
                currentSeparation -= right.offset
                right = getattr(right, "thread", right.left)
            else:
                rightOffsetSum += right.offset
                currentSeparation += right.offset
                right = getattr(right, "thread", right.right)

        # Set the root offset, and include it in the accumulated offsets.
        node.offset = (rootSeparation + 1) / 2
        leftOffsetSum -= node.offset
        rightOffsetSum += node.offset

        # Update right and left extremes
        rightLeftLevel = getattr(rightExtremes.left, "level", -1)
        leftLeftLevel = getattr(leftExtremes.left, "level", -1)
        if rightLeftLevel > leftLeftLevel or not node.left:
            extremes.left = rightExtremes.left
            if extremes.left:
                extremes.left.offset += node.offset

        else:
            extremes.left = leftExtremes.left
            if extremes.left:
                extremes.left.offset -= node.offset

        leftRightLevel = getattr(leftExtremes.right, "level", -1)
        rightRightLevel = getattr(rightExtremes.right, "level", -1)
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
            rightExtremes.right.offset = abs(
                rightExtremes.right.offset + node.offset - leftOffsetSum
            )
        elif right and right != node.right and leftExtremes and leftExtremes.left:
            leftExtremes.left.thread = right
            leftExtremes.left.offset = abs(
                leftExtremes.left.offset - node.offset - rightOffsetSum
            )

        # Return self
        return self

    # Transform relative to absolute coordinates, and measure the bounds of the tree.
    # Return a measurement of the tree in output units.
    def transform(
        self,
        node: BinaryTreeNode,
        x=0,
        unitMultiplierX=1,
        unitMultiplierY=1,
        measure=None,
    ) -> TreeMeasurement:
        if measure is None:
            measure = TreeMeasurement()
        if not node:
            return measure

        node.x = x * unitMultiplierX
        node.y *= unitMultiplierY
        self.transform(
            node.left, x - node.offset, unitMultiplierX, unitMultiplierY, measure
        )
        self.transform(
            node.right, x + node.offset, unitMultiplierX, unitMultiplierY, measure
        )
        if measure.minY > node.y:
            measure.minY = node.y

        if measure.maxY < node.y:
            measure.maxY = node.y

        if measure.minX > node.x:
            measure.minX = node.x

        if measure.maxX < node.x:
            measure.maxX = node.x

        measure.width = abs(measure.minX - measure.maxX)
        measure.height = abs(measure.minY - measure.maxY)
        measure.centerX = measure.minX + measure.width / 2
        measure.centerY = measure.minY + measure.height / 2
        return measure
