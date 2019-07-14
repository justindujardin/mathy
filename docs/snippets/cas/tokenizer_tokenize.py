from typing import List
from mathy.core.tokenizer import Tokenizer, Token

text = "4x + 2x^3 * 7x"
tokenizer = Tokenizer()
tokens: List[Token] = tokenizer.tokenize(text)

for token in tokens:
    print(f"type: {token.type}, value: {token.value}")
