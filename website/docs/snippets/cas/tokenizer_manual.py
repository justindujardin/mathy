from typing import List

from mathy_core import Token, TOKEN_TYPES, Tokenizer

manual_tokens: List[Token] = [
    Token("4", TOKEN_TYPES.Constant),
    Token("x", TOKEN_TYPES.Variable),
    Token("+", TOKEN_TYPES.Plus),
    Token("2", TOKEN_TYPES.Constant),
    Token("", TOKEN_TYPES.EOF),
]
auto_tokens: List[Token] = Tokenizer().tokenize("4x + 2")

for i, token in enumerate(manual_tokens):
    assert auto_tokens[i].value == token.value
    assert auto_tokens[i].type == token.type
