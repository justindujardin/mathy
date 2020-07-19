# mathy.core.tokenizer

## Tokenizer
```python
Tokenizer(self, exclude_padding:bool=True)
```
The Tokenizer produces a list of tokens from an input string.
### eat_token
```python
Tokenizer.eat_token(self, context:mathy.core.tokenizer.TokenContext, typeFn)
```
Eat all of the tokens of a given type from the front of the stream
until a different type is hit, and return the text.
### identify_alphas
```python
Tokenizer.identify_alphas(self, context:mathy.core.tokenizer.TokenContext) -> int
```
Identify and tokenize functions and variables.
### identify_constants
```python
Tokenizer.identify_constants(
    self,
    context: mathy.core.tokenizer.TokenContext,
) -> int
```
Identify and tokenize a constant number.
### identify_operators
```python
Tokenizer.identify_operators(
    self,
    context: mathy.core.tokenizer.TokenContext,
) -> bool
```
Identify and tokenize operators.
### is_alpha
```python
Tokenizer.is_alpha(self, c:str) -> bool
```
Is this character a letter
### is_number
```python
Tokenizer.is_number(self, c:str) -> bool
```
Is this character a number
### tokenize
```python
Tokenizer.tokenize(
    self,
    buffer: str,
    terms = False,
) -> List[mathy.core.tokenizer.Token]
```
Return an array of `Token`s from a given string input.
This throws an exception if an unknown token type is found in the input.
