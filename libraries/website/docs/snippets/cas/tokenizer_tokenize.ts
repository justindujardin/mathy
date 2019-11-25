import { Tokenizer, Token } from "mathy/core/tokenizer"

const text = "4x + 2x^3 * 7x"
const tokenizer = new Tokenizer()
const tokens: Token[] = tokenizer.tokenize(text)
for (token in tokens) {
  console.log(`type: ${token.type}, value: ${token.value}`)
}
