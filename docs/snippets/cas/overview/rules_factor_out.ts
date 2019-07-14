import { DistributiveFactorOutRule } from "mathy/core/rules"
import { ExpressionParser } from "mathy/core/parser"

const input = "4x + 2x"
const output = "(4 + 2) * x"
const parser = new ExpressionParser()

const input_exp = parser.parse(input)
const output_exp = parser.parse(output)

// Verify that the rule transforms the tree as expected
const change = new DistributiveFactorOutRule().apply_to(input_exp)
if (`${change.result}` !== output) {
  throw new Error("invalid output")
}

// Verify that both tress evaluate to the same value
const ctx = { x: 3 }
if (input_exp.evaluate(ctx) !== output_exp.evaluate(ctx)) {
  throw new Error("invalid result")
}
