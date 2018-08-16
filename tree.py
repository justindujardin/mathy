from mathzero.core.layout import TreeLayout
from mathzero.core.parser import ExpressionParser
from mathzero.core.expressions import BinaryExpression, UnaryExpression

parser = ExpressionParser()
node = parser.parse("2x^4 * 4x^3 + (4 / 8) * x")
renderer = TreeLayout()


def print_node(node):
    if isinstance(node, (BinaryExpression, UnaryExpression)):
        return node.name
    return str(node)


renderer.render_curses(node, print_node)
