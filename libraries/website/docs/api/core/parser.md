# mathy.core.parser

## TokenSet
```python
TokenSet(self, source:Union[_ForwardRef('TokenSet'), int])
```
TokenSet objects are bitmask combinations for checking to see
if a token is part of a valid set.
### add
```python
TokenSet.add(self, addTokens)
```
Add tokens to self set and return a TokenSet representing
their combination of flags.  Value can be an integer or an instance
of `TokenSet`
### contains
```python
TokenSet.contains(self, type:int) -> bool
```
Returns true if the given type is part of this set
## ExpressionParser
```python
ExpressionParser(self)
```
Parser for converting text into binary trees. Trees encode the order of
operations for an input, and allow evaluating it to detemrine the expression
value.

### Grammar Rules

Symbols:
```
( )    == Non-terminal
{ }*   == 0 or more occurrences
{ }+   == 1 or more occurrences
{ }?   == 0 or 1 occurrences
[ ]    == Mandatory (1 must occur)
|      == logical OR
" "    == Terminal symbol (literal)
```

Non-terminals defined/parsed by Tokenizer:
```
(Constant) = anything that can be parsed by `float(in)`
(Variable) = any string containing only letters (a-z and A-Z)
```

Rules:
```
(Function)     = [ functionName ] "(" (AddExp) ")"
(Factor)       = { (Variable) | (Function) | "(" (AddExp) ")" }+ { { "^" }? (UnaryExp) }?
(FactorPrefix) = [ (Constant) { (Factor) }? | (Factor) ]
(UnaryExp)     = { "-" }? (FactorPrefix)
(ExpExp)       = (UnaryExp) { { "^" }? (UnaryExp) }?
(MultExp)      = (ExpExp) { { "*" | "/" }? (ExpExp) }*
(AddExp)       = (MultExp) { { "+" | "-" }? (MultExp) }*
(EqualExp)     = (AddExp) { { "=" }? (AddExp) }*
(start)        = (EqualExp)
```

### parse
```python
ExpressionParser.parse(self, input)
```
Parse a string representation of an expression into a tree
that can be later evaluated.

Returns : The evaluatable expression tree.

### next
```python
ExpressionParser.next(self)
```
Assign the next token in the queue to `self.current_token`.

Return True if there are still more tokens in the queue, or False if there
are no more tokens to look at.
### eat
```python
ExpressionParser.eat(self, type)
```
Assign the next token in the queue to current_token if its type
matches that of the specified parameter. If the type does not match,
raise a syntax exception.

Args:
    - `type` The type that your syntax expects @current_token to be

### check
```python
ExpressionParser.check(self, tokens)
```
Check if the `self.current_token` is a member of a set Token types

Args:
    - `tokens` The set of Token types to check against

`Returns` True if the `current_token`'s type is in the set else False
