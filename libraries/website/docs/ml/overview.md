## Overview

Mathy uses machine learning (or ML) to choose which [actions](/cas/rules/overview) to apply to which nodes in an expression tree in order to accomplish [a desired task](/envs/overview).

## Input Preprocessing

Mathy preprocesses the input problem by parsing its text into a tree, converting the tree into a sequence features for each node in the tree, and [...].

### Text to Trees

A problem text is [parsed into a tree](/cas/parser) that encodes the order of operations while removing parentheses and whitespace.
Consider the tree that results from the input: `-3 * (4 + 7)`

`mathy:-3 * (4 + 7)`

!!! info "Order of Operations"

        Remember that the order of operations normally puts higher priority on multiplication than addition, but in this
        case the addition of `4 + 7` has to be resolved first because it is explicitly grouped with parentheses.

### Trees to Lists

Rather than try to feed [expression trees](/cas/parser) into a machine learning model, we [traverse them](/api/core/expressions/#to_list) to produce feature input sequences.

Consider the tree from the previous example encoded in list form: `-3 * (4 + 7)`

`features:-3 * (4 + 7)`

- The first row is the input token characters stripped of whitespace and parentheses.
- The second row is the sequence of floating point values for the tree, with each non-constant node represented by a mask value.
- The third row is the node type integer representing the class of the node in the tree.

### Lists to Observations

A set of feature lists for an expression is cool, but it represents only a single moment in time, so it's not a useful if your task is to transform a tree over time into a target state. To do that you need a concept of time.

#### Example

Consider that you h

### Observations to Embeddings

2. [embedded](/ml/math_embeddings) into variable length sequences of fixed size vectors that can be fed into a Sequence-to-Sequence ML model.
3. [Parse](/cas/parser) the token list into an Expression tree

##

Solving complex math problems requires combinations of low-level actions that lead to higher-level ones. Humans are smart, so we often do multiple steps at once in our heads and consider it to be just one. For example, we think of of the transformation `4x + 2x => 6x` a single action but it actually requires a factoring operation and an artithmetic operation to accomplish.

When it comes to solving math problems, there are at least two broad ways to approach combining low-level rules into high-level actions. Some CAS systems write a ton of custom lower-level rule compositions to explicitly capture the higher-level actions in a usable form. This is extremely effective and can yield systems that are able to solve many different types of math problems with confidence, but it comes at the cost of requiring expert knowledge to craft all the specific problem-set rules. Mathy uses Machine Learning (ML) to pick combinations of low-level rules, and relies on the ML model to put them together to form higher-level actions. This has the benefit of not necessarily requiring expert knowledge, but adds the complexity of crafting an ML model that can pick reasonable sets of actions for many types of problems.
