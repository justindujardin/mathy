from .tree import LEFT, RIGHT
import curses


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
    """Implement a Reingold-Tilford 'tidier' tree layout algorithm."""

    # Assign x/y values to all nodes in the tree, and return an object containing
    # the measurements of the tree.
    def layout(self, node, unitMultiplierX=1, unitMultiplierY=1):
        self.measure(node)
        return self.transform(node, 0, unitMultiplierX, unitMultiplierY)

    # Computer relative tree node positions
    def measure(self, node, level=0, extremes: TidierExtreme = None):
        if extremes == None:
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
    def transform(self, node, x=0, unitMultiplierX=1, unitMultiplierY=1, measure=None):
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

    def render_curses(self, expression, node_namer_fn=None):
        """
        Render an expression tree to the terminal using curses given a function
        that is called with an expression and returns a string that should be
        rendered to represent it. If no node namer is provided, the node will be
        cast to a string type.
        """
        # TODO: Curses sucks for what I'm doing. Allocate a 2d grid initialized 
        #       to spaces and render into that. Then the result can be printed 
        #       like everything else in the world.
        tidier = TreeLayout()
        measure = tidier.layout(expression, 6, 2)

        def wrapper(screen):
            screen.clear()
            curses.curs_set(0)

            def visit_fn(node, depth, data):
                nonlocal measure, node_namer_fn
                text = str(node) if node_namer_fn is None else node_namer_fn(node)
                x = node.x
                if measure.minX < 0:
                    x += abs(measure.minX)
                y = node.y + 1
                if node.parent is not None:
                    cur_x = int(x)
                    if node.parent.getSide(node) == LEFT:
                        target_x = (node.parent.x + abs(measure.minX)) - 2
                        cur_x += 1
                        screen.addstr(int(y) - 1, cur_x, "/")
                        cur_x += 1
                        while cur_x < target_x:
                            screen.addstr(int(y) - 2, cur_x + 1, "_")
                            cur_x += 1
                        
                    else:
                        target_x = (node.parent.x + abs(measure.minX)) + 2
                        cur_x -= 1
                        screen.addstr(int(y) - 1, cur_x, "\\")
                        cur_x -= 1
                        while cur_x > target_x:
                            screen.addstr(int(y) - 2, cur_x - 1, "_")
                            cur_x -= 1

                screen.addstr(int(y), int(x), text)

            expression.visitPostorder(visit_fn)
            screen.refresh()
            screen.getkey()

        curses.wrapper(wrapper)

