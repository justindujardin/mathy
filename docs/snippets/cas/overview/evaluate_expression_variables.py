from mathy import ExpressionParser
from mathy import MathExpression

expression: MathExpression = ExpressionParser().parse("4x + 2y")
assert expression.evaluate({"x": 2, "y": 5}) == 18
