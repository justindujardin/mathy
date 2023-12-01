Mathy's [Computer Algebra System](/cas/overview) creates beautiful trees from math expressions. To modify trees, Mathy exposes a rules-based transformation system.

## Overview

Rules-based systems are known for their flexibility and can have tens or hundreds of rules that define how a system should act.

Contrary to that, Mathy tries to implement as few rules as possible. To that end, it builds rules based on the **Properties of Numbers**, and the agents must combine them in exciting ways to accomplish high-level actions.

!!! info "Properties of Numbers"

    So, do you remember what the properties of numbers are? Or you may want a refresher. Dummies.com has a [decent cheat sheet](https://www.dummies.com/education/math/calculus/understanding-the-properties-of-numbers/){target=\_blank}

## Basic Example

To see how learning from first principles works, consider adding two like terms together: `4x + 2x`

Rather than write a rule for combining like terms, Mathy agents must use multiple more straightforward rules.

Specifically, the [distributive property](/rules/distributive_property_factor_out) is used to factor out the common `x` variable:

| Input = **4x + 2x** | Output = **(4 + 2) \* x** |
| :-----------------: | :------------------------ |
|   `mathy:4x + 2x`   | `mathy:(4 + 2) * x`       |

This leaves the `4 + 2` to be added using [constant arithmetic](/rules/constant_arithmetic).

| Input = **(4 + 2) \* x** | Output = **6x** |
| :----------------------: | :-------------- |
|   `mathy:(4 + 2) * x`    | `mathy:6x`      |

It can help to think of this as challenging the model to learn a new skill.

Mathy has to **learn how to deploy two actions in a sequence** to combine like terms.

## What Rules Do

Rules have two primary responsibilities:

1. Determine if the rule can be applied to a given node in an expression
2. Apply a transformation to an applicable node and return information about what has changed

## Visual Example

Consider the application of constant arithmetic to simplify the expression `12 * 10 + x` so that it results in a tree of `120 + x`

|       Before        | After           |
| :-----------------: | :-------------- |
| `mathy:12 * 10 + x` | `mathy:120 + x` |

The constant simplify rule takes the two constant nodes `12` and `10`, then evaluates them and replaces them inside the tree with a single constant value of `120`

!!! info find_nodes

        When you want to apply a rule to a tree, the `rule.find_nodes` helper can be used to return a list of nodes to which the current rule can be applied.

## Code Example

Consider an example where we want to swap the position of two nodes so that like terms are adjacent to one another for combination:

```Python
{!./snippets/rules/commutative_swap.py!}
```

After using commutative swap, the tree is now arranged with like terms being adjacent, which enables the application of other rules like the [distributive factor out](/rules/distributive_property_factor_out).
