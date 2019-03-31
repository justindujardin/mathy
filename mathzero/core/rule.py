from .tree import STOP
from .util import is_debug_mode

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
        # Only double-check canApply in debug mode
        if is_debug_mode() and not self.canApplyTo(node):
            print("Bad Apply: {}".format(node))
            print("     Root: {}".format(node.getRoot()))
            raise Exception("Cannot apply {} to {}".format(self.name, node))

        return ExpressionChangeRule(self, node)


# Basic description of a change to an expression tree
class ExpressionChangeRule:
    def __init__(self, rule, node=None):
        self.rule = rule
        self.node = node
        self._saveParent = None
        self.focus_node = None

    def set_focus(self, node):
        """Specify the node that is desirable to focus on based on the 
        change that a specific rule has made to a complex tree."""
        self.focus_node = node

    def saveParent(self, parent=None, side=None):
        if self.node and parent is None:
            parent = self.node.parent

        self._saveParent = parent
        if self._saveParent:
            self._saveSide = side or parent.getSide(self.node)

        return self

    def done(self, node):
        if self._saveParent:
            self._saveParent.setSide(node, self._saveSide)
        self.result = node
        return self
