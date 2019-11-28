from typing import List
from mathy import Token, TokenConstant, TokenEOF, Tokenizer, TokenPlus, TokenVariable

manual_tokens: List[Token] = [
    Token("4", TokenConstant),
    Token("x", TokenVariable),
    Token("+", TokenPlus),
    Token("2", TokenConstant),
    Token("", TokenEOF),
]
auto_tokens: List[Token] = Tokenizer().tokenize("4x + 2")

for i, token in enumerate(manual_tokens):
    assert auto_tokens[i].value == token.value
    assert auto_tokens[i].type == token.type
