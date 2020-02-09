## mathy_pydoc

**IMPORTANT** This is a fork of the pydoc_markdown repo to use the legacy version for Mathy's simple API doc needs. See the official repo for the latest stable updates: https://github.com/NiklasRosenstein/pydoc-markdown

&ndash; _insipired by the [Keras] Documentation_

## Installation

    pip install mathy_pydoc

## Usage

mathy_pydoc generates plain Markdown files from Python modules using the
`mathy_pydoc` command. Specify one or more module names on the command-line.
Supports the `+` syntax to include members of the module (or `++` to include
members of the members, etc.)

    mathy_pydoc mypackage+ mypackage.mymodule+ > docs.md

## Syntax

### Cross-references

Symbols in the same namespace may be referenced by using a hash-symbol (`#`)
directly followed by the symbols' name, including relative references. Note that
using parentheses for function names is encouraged and will be ignored and
automatically added when converting docstrings. Examples: `#ClassName.member` or
`#mod.function()`.

For absolute references for modules or members in names that are not available
in the current global namespace, `#::mod.member` must be used (note the two
preceeding two double-colons).

For long reference names where only some part of the name should be displayed,
the syntax `#X~some.reference.name` can be used, where `X` is the number of
elements to keep. If `X` is omitted, it will be assumed 1. Example:
`#~some.reference.name` results in only `name` being displayed.

In order to append additional characters that are not included in the actual
reference name, another hash-symbol can be used, like `#Signal#s`.

### Sections

Sections can be generated with the Markdown `# <Title>` syntax. It is important
to add a whitespace after the hash-symbol (`#`), as otherwise it would represent
a cross-reference. Some special sections alter the rendered result of their
content, including

- Arguments (1)
- Parameters (1)
- Attributes (1)
- Members (1)
- Raises (2)
- Returns (2)

(1): Lines beginning with `<ident> [(<type>[, ...])]:` are treated as
argument/parameter or attribute/member declarations. Types listed inside the
parenthesis (optional) are cross-linked, if possible. For attribute/member
declarations, the identifier is typed in a monospace font.

(2): Lines beginning with `<type>[, ...]:` are treated as raise/return type
declarations and the type names are cross-linked, if possible.

Lines following a name's description are considered part of the most recent
documentation unless separated by another declaration or an empty line. `<type>`
placeholders can also be tuples in the form `(<type>[, ...])`.

### Code Blocks

GitHub-style Markdown code-blocks with language annotations can be used.

    ```python
    >>> for i in range(100):
    ...
    ```

---
