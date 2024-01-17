from mathy_core import ExpressionParser
from mathy_core.rules import DistributiveFactorOutRule

input = "4x + 2x"
output = "(4 + 2) * x"
parser = ExpressionParser()

input_exp = parser.parse(input)
output_exp = parser.parse(output)

# Verify that the rule transforms the tree as expected
change = DistributiveFactorOutRule().apply_to(input_exp)
assert str(change.result) == output

# Verify that both trees evaluate to the same value
ctx = {"x": 3}
assert input_exp.evaluate(ctx) == output_exp.evaluate(ctx)
