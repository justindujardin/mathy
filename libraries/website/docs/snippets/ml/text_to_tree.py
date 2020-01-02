from typing import List

from mathy import ExpressionParser, MathExpression, Token, VariableExpression

problem = "4 + 2x"
parser = ExpressionParser()
tokens: List[Token] = parser.tokenize(problem)
expression: MathExpression = parser.parse(problem)
assert len(expression.find_type(VariableExpression)) == 1
