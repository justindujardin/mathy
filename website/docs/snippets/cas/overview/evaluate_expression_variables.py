from mathy_core import ExpressionParser, MathExpression

expression: MathExpression = ExpressionParser().parse("4x + 2y")
assert expression.evaluate({"x": 2, "y": 5}) == 18
