## Motivation

In order to parse math text into tree structures that encode the order of operations of the input, we first need an intermediate representation. Specifically we want to build a list of characters in the text that correspond to relevant `tokens` in a math expression. That is what the Tokenizer does.

The tokenization process treats the input string as an array of characters, iterating over them to produce an array of tokens that have `type`/`value` properties. While building the array, the tokenizer also checks to be sure that the expression is full of valid math tokens, and discards extra whitespace characters.

## Visual Example

As an example, consider the input text `8 - (2 + 4)` and its token representation.

`tokens:8 - (2 + 4)`

- The top row contains the token value.
- The bottom row contains the integer type of the token represented by the value.

## Code Example

Basic tokenization only requires a few lines of code:

```Python

{!./snippets/cas/tokenizer_tokenize.py!}

```

## Conceptual Example

To better understand the tokenizer, let's build a tokens array manually and compare it to the one that the tokenizer outputs:

```Python
{!./snippets/cas/tokenizer_manual.py!}
```
