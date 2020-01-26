from mathy import CommutativeSwapRule, ExpressionParser

input = "x + y + x"
output = "x + x + y"
parser = ExpressionParser()

input_exp = parser.parse(input)
output_exp = parser.parse(output)

# Verify that the rule transforms the tree as expected
change = CommutativeSwapRule().apply_to(input_exp)
assert str(change.result) == output
