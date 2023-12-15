## Motivation

We first need an intermediate representation to parse math text into tree structures that encode the Order of Operations of the input. Specifically, we want to build a list of text characters corresponding to relevant `tokens` for a math expression. That is what the tokenizer does.

The tokenization process treats the input string as an array of characters, iterating over them to produce a list of tokens with `type`/`value` properties. While building the collection, the tokenizer also optionally discards extra whitespace characters.

## Visual Example

For example, consider the input text `8 - (2 + 4)` and its token representation.

`tokens:8 - (2 + 4)`

- The top row contains the token value.
- The bottom row includes the integer type of the token represented by the value.

## Code Example

Simple tokenization only requires a few lines of code:

```Python

{!./snippets/cas/tokenizer_tokenize.py!}

```

## Conceptual Example

To better understand the tokenizer, let's build a tokens array manually then compare it to the tokenizer outputs:

```Python
{!./snippets/cas/tokenizer_manual.py!}
```
