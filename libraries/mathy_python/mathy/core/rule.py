from typing import Optional, List
from ..core.expressions import MathExpression
from ..core.tree import STOP
from ..util import is_debug_mode


class BaseRule:
    """Basic rule class that visits a tree with a specified visit order."""

    @property
    def name(self):
        """Readable rule name used for debug rendering and description outputs"""
        return "Abstract Base Rule"

    @property
    def code(self) -> str:
        """Short code for debug rendering. Should be two letters."""
        return "XX"

    def find_node(self, expression: MathExpression) -> Optional[MathExpression]:
        """Find the first node that can have this rule applied to it."""
        result = None

        def visit_fn(node, depth, data):
            nonlocal result
            if self.can_apply_to(node):
                result = node

            if result is not None:
                return STOP

        expression.visit_inorder(visit_fn)
        return result

    def find_nodes(self, expression: MathExpression) -> List[MathExpression]:
        """Find all nodes in an expression that can have this rule applied to them.
        Each node is marked with it's token index in the expression, according to
        the visit strategy, and stored as `node.r_index` starting with index 0
        """
        nodes = []
        index = 0

        def visit_fn(node, depth, data):
            nonlocal nodes, index
            add = None
            node.r_index = index
            if self.can_apply_to(node):
                add = node

            index += 1
            if add:
                return nodes.append(add)

        expression.visit_inorder(visit_fn)
        return nodes

    def can_apply_to(self, node: MathExpression) -> bool:
        """User-specified function that returns True/False if a rule can be
        applied to a given node.

        !!!warning "Performance Point"

            `can_apply_to` is called very frequently during normal operation
            and should be implemented as efficiently as possible.
        """
        return False

    def apply_to(self, node: MathExpression) -> "ExpressionChangeRule":
        """Apply the rule transformation to the given node, and return a
        ExpressionChangeRule object that captures the input/output states
        for the change."""
        # Only double-check canApply in debug mode for performance reasons
        if is_debug_mode() and not self.can_apply_to(node):
            print("Bad Apply: {}".format(node))
            print("     Root: {}".format(node.get_root()))
            raise Exception("Cannot apply {} to {}".format(self.name, node))

        return ExpressionChangeRule(self, node)


class ExpressionChangeRule:
    """Object describing the change to an expression tree from a rule transformation"""

    rule: BaseRule
    node: Optional[MathExpression]
    result: Optional[MathExpression]
    _save_parent: Optional[MathExpression]

    def __init__(self, rule, node: MathExpression = None):
        self.rule = rule
        self.node = node
        self.result = None
        self._save_parent = None

    def save_parent(
        self, parent: MathExpression = None, side: Optional[str] = None
    ) -> "ExpressionChangeRule":
        """Note the parent of the node being modified, and set it as the parent of the
        rule output automatically."""
        if self.node and parent is None:
            parent = self.node.parent

        self._save_parent = parent
        if parent:
            self._save_side = side or parent.get_side(self.node)

        return self

    def done(self, node) -> "ExpressionChangeRule":
        """Set the result of a change to the given node. Restore the parent
        if `save_parent` was called."""
        if self._save_parent:
            self._save_parent.set_side(node, self._save_side)
        self.result = node
        return self
