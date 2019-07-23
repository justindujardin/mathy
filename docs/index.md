Mathy is a library for parsing and transforming math problems from arbitrary input states into desired output states. It uses Reinforcement Learning (a form of machine learning, or artificial intelligence) build an understanding of how to effectively solve math problems step-by-step without programming the system with heuristics about which rules to apply in what order.


---

Mathy really wants to help you with your homework. It is limited to only a few types of problems currently, but it's open source and
extensible, so you can add support for solving new types of problems!

<!-- Mathtastic is a Computer Algebra System and Machine Learning environment for building Reinforcement Learning agents that can manipulate math and show their work step-by-step. The stated goal is to make an open source math tutoring suite that is available to every human that has access to a computer. Today I want to tell you about the first step along the path to that lofty goal: simplifying polynomial expressions. -->

# How it works

Let's say you have an input: `4x + 2x`

First Mathtastic loops over the input string to make an array of tokens, one for each character:

```
[("CONST", 4), ("VAR", "x"), ("PLUS","+"), ("CONST", 2), ("CONST", "x")]
```

With a list of tokens in hand, we parse them into a tree. This step fills in the missing context from our token arrays (e.g. the implied multiplication between 4 and x,) verifies the correctness of the token stream (e.g. not missing closing parenthesis,) and encodes the order of operations by outputting a tree structure. Keeping with our text representation, the output looks like this:

```
PLUS(
  MULT( CONST(4), VAR("x") ),
  MULT( CONST(2), VAR("x") )
)
```

Now that the expression is represented in binary tree[link] form we can manipulate it in interesting ways to produce different trees that have the same value as the input.

If you're starting to doze off, stay with me a bit longer, this is where things start to get cool. Let's render our tree structure to get a better sense of what it looks like:

<MathText input="4x * p * p * 12x^2" />
<MathTree input="4x * p * p * 12x^2" />
<MathText input="4x * 4p * 4p * 12x^2" />
<MathTree input="4x * 4p * 4p * 12x^2" />
<MathText input="4x * p^2 * p^3 * 12x^2" />
<MathTree input="4x * p^2 * p^3 * 12x^2" />
<MathText input="4x * 4p^2 * p^3 * 12x^2" />
<MathTree input="4x * 4p^2 * p^3 * 12x^2" />

While it's great to have a pretty tree visualization and know that expressions are valid math, that's not itself very useful. What we'd like is to "output" a new tree from an "input" using some kind of "rules" that ensure that the transformations don't change the meaning of the input expression. Said another way, we want to change the expressions so that "4 _ 5" can be changed to "5 _ 4" but not "4 + 5". Enter the Mathy rules system...

# Transformation [Rules](https://mathy.ai)

Parsed MathExpression trees can be manipulated using "rules" that take an input expression and return an output tree that is equivalent to the input but stated in a different way. A basic example might switch the order of two terms separated by an addition, from "2 + 4" to "4 + 2". This change is valid because the same value is obtained by evaluating the input and output expressions.

As you may have already guessed, the base rules defined for Mathy map directly to the "Properties of Numbers" from algebra: the commutative property allows the example above, while the associative property changes the evaluation order of nodes in the tree ("4 + 2x + 3x" to "4 + (2x + 3x)" [<--check that example], and the distributive property allows factoring out common elements from nearby terms ("4x + 4x" to "(4 + 4) _ x") as well as redistributing nodes back after factoring them out ("x _ (4 +4)" to "4x + 4x").

Mathy knows about actions that map to mathematical rules (axioms?) such that it can only interact with expressions in primitive ways. While it's possible to construct complex chains of actions using heuristics, Mathy prefers to trade that complexity for a machine learning based approach to finding optimal tree transformations to end up in the desired output representation.

Rules define two main methods, one to determine if they can be applied to an expression _at a given node_, and another to apply the rule to a node and return a new tree.

Coming back to our original example of "4x + 2x" that we parsed into a tree of nodes

```
PLUS(
  MULT( CONST(4), VAR("x") ),
  MULT( CONST(2), VAR("x") )
)
```

Traversing trees is a bit more difficult conceptually because you have to consider the order of traversal, making them more difficult to reason with. As a trick, we collapse the tree into a list form that expresses the hierarchy in a known traversal order. That is, we flatten the trees assuming that they are traversed in a fixed order, so we can store and compare them more easily.

Let's look at our tree from above after being converted to list form:

```
["CONST_4", "MULT", "VAR_X", "PLUS", "CONST_2", "MULT", "VAR_X"]
```

--

# Properties of Numbers

When working with Arithmetic the Order of Operations is crucial, and when working with Algebra the Properties of Numbers are essential. Remember that the properties of numbers are things like the Commutative Property that lets you switch the order of two Add terms without changing the value (e.g. `4x + 2x = 2x + 4x`)

Mathy side-steps the arithmetic and order of operations concerns by using Mathtastic to sanitize the inputs and ensure that they're valid. By ignoring this concern, the agent can focus on the properties of numbers and how they can be used to coerce an expression into a desired target state.

--

# Experiment: Needle in a Haystack

One of the first challenges with training a math agent using Monte Carlo Tree Search (MCTS - http://mcts.ai/) is that as the expressions get larger, the tree search takes more and more time to run rollout simulations. This is not so much of a concern if you're Google and are running an experiment at massive scale, but if you're training a math agent on your laptop it can be a deal breaker.

This experiment tries to force the model to learn how to apply desired properties of numbers transformations to large sequences of nodes without paying for the cost of running rollouts over hundreds of moves and nodes. The key insight here is that the model needs observations of correct "useful" moves performed on very large trees in order to correctly predict an action policy when those nodes are seen in the wild.

The setup is fairly simple, we generate a long string of terms and randomly place a target set of nodes amongst them. From there we set the move limit of the self-play to the minimum number required to perform the operation we want the model to learn.

Let's say we want the agent to learn about how to combine like terms in 3 term polynomials, but we also want it to understand how to apply the same set of rules in 24 term problems.

```
16j^6 + 2g^22 + 20k + 24h + 18f^5 + 19a^22 + b + 11e^7 + 21l^2 + 9x^19 + n^7 + 2i + 3y + 6m^19 + 24o^4 + 10c^12 + 10r^13 + q^17 + p^3 + 24z + z + 5t
```

Why it works
The intuition behind constructing these types of problems has to do with the methods we use for finding optimal answers. Because we use a Tree search that has to traverse until a terminal (end) state, the more moves in a game, the longer the simulations will take to complete. By reducing the number of moves to a very small number it's possible for the MCTS search to find good solutions to the first few moves, regardless of the initial tree complexity. That is, even though there may be 500 nodes to simulate actions on, if the simulation ends after 2 turns, the total complexity is relatively low compared to 500 nodes simulated for 100 or more turns. The other reason for success I suspect has to do with the first few moves being significant in the problem set up. We use Dirichlet noise in the root tree search node that causes the root actions to be explored thoroughly during training. If there is only 1 viable root action in each small problem, the noise will ensure that a tree search finds it quite often. Noisy root search combined with constrained move limit problems is a cheap way to gather positive examples of desired behaviors in variable length sequences.