"""Tree Layout
---

In order to help visualize, understand, and debug math trees and transformations to
them, Mathy implements a
[Reingold-Tilford](https://reingold.co/tidier-drawings.pdf){target=_blank} layout
algorithm that works with expression trees. It produces beautiful trees like:

`mathy:(2x^3 + y)(14 + 2.3y)`

"""
from .tree import BinaryTreeNode


class TreeLayout:
    """Calculate a visual layout for input trees."""

    def layout(
        self,
        node: BinaryTreeNode,
        unit_x_multiplier: float = 1.0,
        unit_y_multiplier: float = 1.0,
    ) -> "TreeMeasurement":
        """Assign x/y values to all nodes in the tree, and return an object containing
        the measurements of the tree.

        Returns a TreeMeasurement object that describes the bounds of the tree"""
        self.measure(node)
        return self.transform(node, 0, unit_x_multiplier, unit_y_multiplier)

    def measure(
        self, node: BinaryTreeNode = None, level=0, extremes: "TidierExtreme" = None,
    ) -> "TreeLayout":
        if extremes is None:
            extremes = TidierExtreme()

        # left and right subtree extreme leaf nodes
        left_extremes = TidierExtreme()
        right_extremes = TidierExtreme()

        # separation at the root of the current subtree and current level.
        current_separation = 0.0
        root_separation = 0.0
        min_separation = 1.0

        # The offset from left/right children to the root of the current subtree.
        left_offset_sum = 0.0
        right_offset_sum = 0.0

        # Avoid selecting as extreme
        if not node:
            if extremes.left is not None:
                extremes.left.level = -1

            if extremes.right is not None:
                extremes.right.level = -1

            return self

        # Assign the `node.y`, note the left/right child nodes, and recurse
        node.y = level
        left = node.left
        right = node.right
        self.measure(left, level + 1, left_extremes)
        self.measure(right, level + 1, right_extremes)

        # A leaf is both the leftmost and rightmost node on the lowest level of the
        # subtree consisting of itself.
        if not node.right and not node.left:
            node.offset = 0
            extremes.right = extremes.left = node
            return self

        # if only a single child, assign the next available offset and return.
        if not node.right or not node.left:
            node.offset = min_separation
            extremes.right = extremes.left = node.left if node.left else node.right
            return self

        # Set the current separation to the minimum separation for the root of the
        # subtree.
        current_separation = min_separation
        left_offset_sum = right_offset_sum = 0

        # Traverse the subtrees until one of them is exhausted, pushing them apart
        # as needed.
        loops = 0
        while left and right:
            loops = loops + 1
            if loops > 100000:
                raise Exception("An impossibly large tree perhaps?")

            if current_separation < min_separation:
                root_separation += min_separation - current_separation
                current_separation = min_separation

            if left.right and left.offset:
                left_offset_sum += left.offset
                current_separation -= left.offset
                left = getattr(left, "thread", left.right)
            elif left.offset is not None:
                left_offset_sum -= left.offset
                current_separation += left.offset
                left = getattr(left, "thread", left.left)

            if right.left and right.offset:
                right_offset_sum -= right.offset
                current_separation -= right.offset
                right = getattr(right, "thread", right.left)
            elif right.offset is not None:
                right_offset_sum += right.offset
                current_separation += right.offset
                right = getattr(right, "thread", right.right)

        # Set the root offset, and include it in the accumulated offsets.
        node.offset = (root_separation + 1) / 2
        left_offset_sum -= node.offset
        right_offset_sum += node.offset

        # Update right and left extremes
        right_left_level = getattr(right_extremes.left, "level", -1)
        left_left_level = getattr(left_extremes.left, "level", -1)
        if right_left_level > left_left_level or not node.left:
            extremes.left = right_extremes.left
            if extremes.left:
                extremes.left.offset += node.offset

        else:
            extremes.left = left_extremes.left
            if extremes.left:
                extremes.left.offset -= node.offset

        left_right_level = getattr(left_extremes.right, "level", -1)
        right_right_level = getattr(right_extremes.right, "level", -1)
        if left_right_level > right_right_level or not node.right:
            extremes.right = left_extremes.right
            if extremes.right:
                extremes.right.offset -= node.offset

        else:
            extremes.right = right_extremes.right
            if extremes.right:
                extremes.right.offset += node.offset

        # If the subtrees have uneven heights, check to see if they need to be
        # threaded.  If threading is required, it will affect only one node.
        if left and left != node.left and right_extremes and right_extremes.right:
            right_extremes.right.thread = left
            right_extremes.right.offset = abs(
                right_extremes.right.offset + node.offset - left_offset_sum
            )
        elif right and right != node.right and left_extremes and left_extremes.left:
            left_extremes.left.thread = right
            left_extremes.left.offset = abs(
                left_extremes.left.offset - node.offset - right_offset_sum
            )

        return self

    def transform(
        self,
        node: BinaryTreeNode = None,
        x=0,
        unit_x_multiplier=1,
        unit_y_multiplier=1,
        measure=None,
    ) -> "TreeMeasurement":
        """Transform relative to absolute coordinates, and measure the bounds of the tree.

        Return a measurement of the tree in output units."""
        if measure is None:
            measure = TreeMeasurement()
        if not node:
            return measure

        node.x = x * unit_x_multiplier
        node.y *= unit_y_multiplier
        self.transform(
            node.left, x - node.offset, unit_x_multiplier, unit_y_multiplier, measure
        )
        self.transform(
            node.right, x + node.offset, unit_x_multiplier, unit_y_multiplier, measure
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


class TidierExtreme:
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

