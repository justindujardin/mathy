from mathy import (
    AddExpression,
    BaseRule,
    ExpressionParser,
    NegateExpression,
    SubtractExpression,
)


class PlusNegationRule(BaseRule):
    """Convert subtract operators to plus negative to allow commuting"""

    @property
    def name(self) -> str:
        return "Plus Negation"

    @property
    def code(self) -> str:
        return "PN"

    def can_apply_to(self, node) -> bool:
        is_sub = isinstance(node, SubtractExpression)
        is_parent_add = isinstance(node.parent, AddExpression)
        return is_sub and (node.parent is None or is_parent_add)

    def apply_to(self, node):
        change = super().apply_to(node)
        change.save_parent()  # connect result to node.parent
        result = AddExpression(node.left, NegateExpression(node.right))
        result.set_changed()  # mark this node as changed for visualization
        return change.done(result)


parser = ExpressionParser()
expression = parser.parse("4x - 2x")
rule = PlusNegationRule()

# Find a node and apply the rule
applicable_nodes = rule.find_nodes(expression)
assert len(applicable_nodes) == 1
assert applicable_nodes[0] is not None

# Verify the expected change
change = rule.apply_to(applicable_nodes[0])
assert str(change.result) == "4x + -2x"
