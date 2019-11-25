## Motivation

In order to parse math text into tree structures that encode the order of operations of the input, we first need an intermediate representation. Specifically we want to build a list of characters in the text that correspond to relevant `tokens` in a math expression. That is what the Tokenizer does.

!!! note
    The tokenization process makes writing an expression parser much easier, because it sanitizes the input string by removing spaces and other irrelevant information.

The tokenization process treats the input string as an array of characters, and iterates over them producing an output array of tokens that have both a `type` and `value` property. While building the array, the tokenizer also 
checks to be sure that the expression is valid math.

## Code Example

Basic tokenization only requires a few lines of code:

``` Python tab="Python"

{!./snippets/cas/tokenizer_tokenize.py!}

```

``` Typescript tab="Typescript"

{!./snippets/cas/tokenizer_tokenize.ts!}

```


## Conceptual Example

To better understand the tokenizer, let's build a tokens array manually and compare it to the one that the tokenizer outputs:

``` Python
{!./snippets/cas/tokenizer_manual.py!}
```

