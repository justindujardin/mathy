Mathy includes what's called a Computer Algebra System (or CAS). Its job is to turn text into math trees that can be examined and manipulated by way of a two-step process:

1. [Tokenize](/cas/tokenizer) the text into a list of `type`/`value` pairs
2. [Parse](/cas/parser) the token list into an Expression tree

## Examples

### Arithmetic

To get a sense of how Mathy's CAS components work, let's add some numbers together and assert that the result is what we think it should be.

```Python
{!./snippets/cas/overview/evaluate_expression.py!}
```

### Variables Evaluation

Mathy can also deal with expressions that have variables.

When an expression has variables in it, you can evaluate it by providing the "context" to use:

```Python
{!./snippets/cas/overview/evaluate_expression_variables.py!}
```

### Tree Transformations

Mathy can also transform the parsed Expression trees using a set of rules that change the tree structure without altering the value it outputs when you call `evaluate()`.

```python

{!./snippets/cas/overview/rules_factor_out.py!}

```
