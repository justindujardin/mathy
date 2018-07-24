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


class FormattedChange:
    def __init__(self, clone, rootClone):
        self.root = rootClone
        self.rootStr = "{}".format(rootClone)
        self.expression = clone
        self.expressionStr = "{}".format(clone)


# Basic description of a change to an expression tree
class ExpressionChangeRule:
    def __init__(self, rule, node=None, end=None):
        self.rule = rule
        self.node = node
        self.children = []
        self.logs = []
        self._saveParent = None
        if node:
            self.init(node)

        if node and end:
            self.done(end)

    def count(self):
        return len(self.children)

    def saveParent(self, parent=None, side=None):
        if self.node and parent == None:
            parent = self.node.parent

        self._saveParent = parent
        if self._saveParent:
            self._saveSide = side or parent.getSide(self.node)

        return self

    def init(self, node):
        self.begin = self._formatChange(node)
        return self

    def done(self, node):
        if self._saveParent:
            self._saveParent.setSide(node, self._saveSide)

        self.end = self._formatChange(node)
        if self.count() == 0:
            self.note(self)
        return self

    def _formatChange(self, node):
        clone = node.rootClone()
        rootClone = clone.getRoot()
        result = FormattedChange(clone, rootClone)
        return result

    def note(self, change):
        return self.logs.append(
            """`{}:\n   {}\n = {}`""".format(
                change.rule.getName(), change.begin.root, change.end.root
            )
        )

    def log(self, change):
        if not isinstance(change, ExpressionChangeRule):
            raise Exception("Unknown change object type")

        self.note(change)
        self.children.append(change)
        return self

