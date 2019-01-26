from .tree import STOP

# Basic rule class that visits a tree with a specified visit order.
class BaseRule:
    @property
    def name(self):
        return "Abstract Base Rule"

    def findNode(self, expression, includeAll=True):
        result = None

        def visit_fn(node, depth, data):
            nonlocal result
            if self.canApplyTo(node):
                result = node

            if result != None:
                return STOP

        expression.visitPreorder(visit_fn)
        return result

    def findNodes(self, expression, includeAll=True):
        """
        Find all nodes in an expression that can have this rule applied to them.
        Each node is marked with it's token index in the expression, according to 
        the visit strategy, and stored as `node.r_index` starting with index 0
        """
        nodes = []
        index = 0

        def visit_fn(node, depth, data):
            nonlocal nodes, index
            add = None
            node.r_index = index
            if self.canApplyTo(node):
                add = node

            index += 1
            if add:
                return nodes.append(add)

        expression.visitPreorder(visit_fn)
        return nodes

    def canApplyTo(self, node):
        return False

    def applyTo(self, node):
        if not self.canApplyTo(node):
            # print('Bad Apply: {}'.format(node))
            # print('     Root: {}'.format(node.getRoot()))
            raise Exception("Cannot apply {} to {}".format(self.name, node))

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
        return """{}:\n    {}\n    {}""".format(
            self.rule.name, self.begin.getRoot(), self.end.getRoot()
        )
