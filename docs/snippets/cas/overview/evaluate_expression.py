from mathy.core.parser import ExpressionParser

expression = ExpressionParser().parse("4 + 2")
assert expression.evaluate() == 6
