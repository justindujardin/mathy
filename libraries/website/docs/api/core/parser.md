# mathy.core.parser

## ExpressionParser
```python
ExpressionParser(self)
```

### parse
```python
ExpressionParser.parse(self, input)
```
Parse a string representation of an expression into a tree
that can be later evaluated.

Returns : The evaluatable expression tree.

### eat
```python
ExpressionParser.eat(self, type)
```
Assign the next token in the queue to current_token if its type
matches that of the specified parameter. If the type does not match,
raise a syntax exception.

Args:
    type The type that your syntax expects @current_token to be

