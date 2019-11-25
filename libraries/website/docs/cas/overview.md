The core component that makes all of Mathy possible is generally called a Computer Algebra Software (CAS) system. The purpose of this system is to turn plain-text representations of math into symbollic ones that can be examined and manipulated.

## Examples

### Arithmetic

To get a sense for how Mathy's CAS components work, let's add some numbers together and assert that the end result is what we think it should be:

```Python
{!./snippets/cas/overview/evaluate_expression.py!}
```

!!! info

        See how the input was a text string, but we make an assertion about the number value it represents? To make this happen Mathy has to:

        1. Tokenize the text
        2. Parse it into an Expression
        3. Evaluate the Expression

        This is the main function of Mathy's CAS system.

### Variables Evaluation

Mathy can also deal with expressions that have variables.

When an expression has variables in it, you can evaluate it by passing the "context" to use:

```Python
{!./snippets/cas/overview/evaluate_expression_variables.py!}
```

### Tree Transformations

Mathy can also transform the parsed Expression trees using a set of Rules that change the tree structure without altering the value it outputs when you call `evaluate()`.

```python

{!./snippets/cas/overview/rules_factor_out.py!}

```
