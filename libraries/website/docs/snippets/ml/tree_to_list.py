from typing import List

from mathy import ExpressionParser, MathExpression

parser = ExpressionParser()
expression: MathExpression = parser.parse("4 + 2x")
nodes: List[MathExpression] = expression.to_list()
# len([4,+,2,*,x])
assert len(nodes) == 5
