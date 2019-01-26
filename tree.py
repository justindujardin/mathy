from mathzero.core.layout import TreeLayout
from mathzero.core.parser import ExpressionParser
from mathzero.core.expressions import BinaryExpression, UnaryExpression

parser = ExpressionParser()
node = parser.parse("(y^2 * 9) * (1 * 7) + 3y")
renderer = TreeLayout()


def print_node(node):
    if isinstance(node, (BinaryExpression, UnaryExpression)):
        return node.name
    return str(node)

#      *
#    /   \
#   2     ^
#       /   \
#      x     4

renderer.render_curses(node, print_node)
