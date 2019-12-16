Mathy's [Computer Algebra System](/cas/overview) creates beautiful trees out of math expressions. To modify trees mathy exposes a rules-based transformation system. Rules have two primary responsibilities:

1.  Determine if the rule [can be applied to](/api/core/rule/#can_apply_to) a given node in an expression
2.  [Apply a transformation](/api/core/rule/#apply_to) to an applicable node, and return information about what has changed

## Visual Example

Consider the application of constant arithmetic to simplify the expression `12 * 10 + x` so that it results in a tree of `120 + x`

|       Before        | After           |
| :-----------------: | :-------------- |
| `mathy:12 * 10 + x` | `mathy:120 + x` |

The constant simplify rule takes the two constant nodes `12` and `10` then evaluates them and replaces them inside the tree with a single constant value of `120`

!!! info find_nodes

        When you want to apply a rule to a tree the [find_nodes](/api/core/rule/#find_nodes) helper can be used to return a list
        of nodes that the current rule can be applied to.

## Code Example

Consider an example where we want to swap the position of two nodes, so that like terms are adjacent to one another for combination:

```Python
{!./snippets/rules/commutative_swap.py!}
```

After using commutative swap, the tree is now arranged with like terms being adjacent, which enables the application of other rules like the [distributive factor out](/rules/distributive_property_factor_out).
