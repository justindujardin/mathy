from ..tree_node import STOP

# Basic rule class that visits a tree with a specified visit order.
class BaseRule:
    def getName(self):
        return "Abstract Base Rule"

    def findNode(self, expression, includeAll=True):
        result = None

        def visit_fn(node, depth, data):
            if includeAll and self.canApplyTo(node):
                result = node
            elif not includeAll and self.shouldApplyTo(node):
                result = node

            if result != None:
                return STOP

        expression.visitPreorder(visit_fn)
        return result

    def findNodes(self, expression, includeAll=True):
        nodes = []

        def visit_fn(node, depth, data):
            add = None
            if includeAll and self.canApplyTo(node):
                add = node
            elif not includeAll and self.shouldApplyTo(node):
                add = node

            if add:
                return nodes.append(add)

        expression.visitPreorder(visit_fn)
        return nodes

    def canApplyTo(self, node):
        return False

    def shouldApplyTo(self, node):
        return self.canApplyTo(node)

    def getWeight(self):
        return 1

    def applyTo(self, node):
        if not self.canApplyTo(node):
            # print('Bad Apply: {}'.format(node))
            # print('     Root: {}'.format(node.getRoot()))
            raise Exception("Cannot apply {} to {}".format(self.getName(), node))

        return ExpressionChangeRule(self, node)


# Basic description of a change to an expression tree
class ExpressionChangeRule:
    def __init__(self, rule, node=None, end=None):
        self.rule = rule
        self.node = node
        self._saveParent = None
        if node:
            self.init(node)

        if node and end:
            self.done(end)

    def saveParent(self, parent=None, side=None):
        if self.node and parent is None:
            parent = self.node.parent

        self._saveParent = parent
        if self._saveParent:
            self._saveSide = side or parent.getSide(self.node)

        return self

    def init(self, node):
        self.begin = node.rootClone()
        return self

    def done(self, node):
        if self._saveParent:
            self._saveParent.setSide(node, self._saveSide)
        self.end = node.rootClone()
        return self

    def describe(self):
        return """`{}:\n   {}\n = {}`""".format(
            self.rule.getName(), self.begin.getRoot(), self.end.getRoot()
        )
