# Simple Solver [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/website/docs/examples/heuristics.ipynb)

> This notebook is built using [mathy_core](https://core.mathy.ai).

Remember the challenges in Algebra of combining like terms to simplify expressions? For example, turning `4x + y + 2x + 14x` into `20x + y` is a fundamental skill in mathematics, useful in various real-world applications like engineering and economics.

While having a program that outputs `20x + y` directly is convenient, understanding the step-by-step transformation is invaluable for learning and problem-solving.

Let's explore how [mathy_core](https://core.mathy.ai) parses input text into a tree structure, and then applies transformations to simplify the tree into a solution. 

You'll see not just the end result but each step we take along the way to get there - a valuable tool for both students and educators alike.


```python
!pip install mathy_core>=0.9.3
```

## Mathy Overview

Before we get started, let's review how mathy core works

1. "4x + 2x" is broken into a list of token, roughly one per character in the input
2. The list of tokens is then parsed into a binary tree structure that can be evaluated and transformed
3. Rules are applied that make changes to the tree structures

We'll use the `ExpressionParser` class to parse the inputs to trees, and some basic built-in rules from `mathy_core.rules`


```python
from typing import List, Optional

from mathy_core import (
    BaseRule,
    ExpressionChangeRule,
    ExpressionParser,
    MathExpression,
    util,
    rules as mathy_rules
)

parser = ExpressionParser()
parser
```




    <mathy_core.parser.ExpressionParser at 0x7f9d040125b0>



## Solution Checking

In order to do more than randomly transform a tree, we need to be able to provide a yes/no answer to whether a given input tree structure is equivalent to what we want for our solution.

In the case of polynomial simplification we just want to check that there are no **like terms** left in the tree, and that the terms that are left are in the preferred arrangement.

> Preferred term arrangement has the coefficient on the left and a variable on the right, e.g. `2x` rather than `x * 2`

Let's implement that function for use in our simplification loop that's coming up. We'll make use of the generous set of utility functions provided by [mathy_core.util](https://core.mathy.ai/api/util) to find the "term nodes" inside our expression, and then verify that there are no like terms in the bunch.


```python
def is_simplified(expression: Optional[MathExpression]) -> bool:
    """If there are no like terms, consider the expression simplified."""
    is_win = False
    # check if there are any like terms
    if expression is None or util.has_like_terms(expression):
        return False
    # check if all terms are in preferred form
    term_nodes: List[MathExpression] = util.get_terms(expression)
    is_win = True
    term: MathExpression
    for term in term_nodes:
        if not util.is_preferred_term_form(term):
            is_win = False
            break
    return is_win


# Let's verify that it does what we expect
assert is_simplified(parser.parse("2x + x")) is False
assert is_simplified(parser.parse("2x + 17y - x")) is False


assert is_simplified(parser.parse("2x + y")) is True
assert is_simplified(parser.parse("2x + y + z + 17q^2")) is True
assert is_simplified(parser.parse("2x^3 + y + 17x")) is True
```

## Transformation Loop

The simplest way to use [mathy_core](https://core.mathy.ai) is by applying transformations to random valid nodes until you reach the desired state. You won't get the optimal path to the solution, and if the problem is complex enough you may not get to the solution, but it's vastly simpler compared to writing more complete heuristics, so here we go.

We'll write a function that takes in a user input, then loops over the parsed expression applying transformations in a certain order, to randomly chosen valid nodes for each rule. The rule ordering is important because it loosely mimics the order in which you would want to do these operations for this type of problem.


```python
import random


def simplify_polynomial(input_text: str, max_steps: int = 10) -> str:
    parser = ExpressionParser()
    expression: Optional[MathExpression] = parser.parse(input_text)
    rules: List[BaseRule] = [
        # 1. Factor out common terms always if we can
        mathy_rules.DistributiveFactorOutRule(),
        # 2. Simplify constants whenever possible
        mathy_rules.ConstantsSimplifyRule(),
        # 3. If we can't perform any of the above, move things
        mathy_rules.CommutativeSwapRule(preferred=False),
    ]
    steps = 0
    last_action = "input"
    print(f"STEP[0]: {last_action:<25} | {expression}")
    while not is_simplified(expression) and steps < max_steps:
        steps += 1
        for rule in rules:
            options = rule.find_nodes(expression)
            if len(options) == 0:
                continue
            option = random.choice(options)
            change: ExpressionChangeRule = rule.apply_to(option)
            assert change.result is not None, "result should not be None"
            expression = change.result.get_root()
            last_action = rule.name
            break
        print(f"STEP[{steps}]: {last_action:<25} | {expression}")

    # Print the final result
    outcome = "WIN" if is_simplified(expression) else "LOSE"
    print(f"FINAL: {expression} ---- {outcome}")
    return str(expression)
```

## Results

Now that we have a function for simplifying polynomials, we can invoke it to see a step-by-step solution. For more complex problems you may need more steps.


```python
simplify_polynomial("4x + y + 2x + 14x")
simplify_polynomial("4j + y + 2p + 14x + 2y + 3x + 7p + 8y + 9j + 10y", max_steps=100)
```

    STEP[0]: input                     | 4x + y + 2x + 14x
    STEP[1]: Distributive Factoring    | 4x + y + (2 + 14) * x
    STEP[2]: Constant Arithmetic       | 4x + y + 16x
    STEP[3]: Commutative Swap          | 4x + 16x + y
    STEP[4]: Distributive Factoring    | (4 + 16) * x + y
    STEP[5]: Constant Arithmetic       | 20x + y
    FINAL: 20x + y ---- WIN
    STEP[0]: input                     | 4j + y + 2p + 14x + 2y + 3x + 7p + 8y + 9j + 10y
    STEP[1]: Commutative Swap          | 4j + y + 2p + 14x + 3x + 2y + 7p + 8y + 9j + 10y
    STEP[2]: Distributive Factoring    | 4j + y + 2p + (14 + 3) * x + 2y + 7p + 8y + 9j + 10y
    STEP[3]: Constant Arithmetic       | 4j + y + 2p + 17x + 2y + 7p + 8y + 9j + 10y
    STEP[4]: Commutative Swap          | 4j + y + 2p + 17x + 2y + 7p + 9j + 8y + 10y
    STEP[5]: Distributive Factoring    | 4j + y + 2p + 17x + 2y + 7p + 9j + (8 + 10) * y
    STEP[6]: Constant Arithmetic       | 4j + y + 2p + 17x + 2y + 7p + 9j + 18y
    STEP[7]: Commutative Swap          | 4j + 2p + y + 17x + 2y + 7p + 9j + 18y
    STEP[8]: Commutative Swap          | 4j + y + 2p + 17x + 2y + 7p + 9j + 18y
    STEP[9]: Commutative Swap          | 4j + y + 2p + 17x + 2y + 9j + 7p + 18y
    STEP[10]: Commutative Swap          | 4j + y + 17x + 2p + 2y + 9j + 7p + 18y
    STEP[11]: Commutative Swap          | 4j + y + 17x + 2p + 2y + 9j + 18y + 7p
    STEP[12]: Commutative Swap          | 4j + y + 17x + 2p + 2y + 18y + 9j + 7p
    STEP[13]: Distributive Factoring    | 4j + y + 17x + 2p + (2 + 18) * y + 9j + 7p
    STEP[14]: Constant Arithmetic       | 4j + y + 17x + 2p + 20y + 9j + 7p
    STEP[15]: Commutative Swap          | y + 4j + 17x + 2p + 20y + 9j + 7p
    STEP[16]: Commutative Swap          | y + 4j + 17x + 2p + 9j + 20y + 7p
    STEP[17]: Commutative Swap          | y + 4j + 17x + 2p + 20y + 9j + 7p
    STEP[18]: Commutative Swap          | y + 4j + 17x + 20y + 2p + 9j + 7p
    STEP[19]: Commutative Swap          | y + 17x + 4j + 20y + 2p + 9j + 7p
    STEP[20]: Commutative Swap          | y + 17x + 4j + 20y + 2p + 7p + 9j
    STEP[21]: Distributive Factoring    | y + 17x + 4j + 20y + (2 + 7) * p + 9j
    STEP[22]: Constant Arithmetic       | y + 17x + 4j + 20y + 9p + 9j
    STEP[23]: Commutative Swap          | y + 4j + 17x + 20y + 9p + 9j
    STEP[24]: Commutative Swap          | y + 4j + 20y + 17x + 9p + 9j
    STEP[25]: Commutative Swap          | y + 4j + 20y + 17x + 9j + 9p
    STEP[26]: Commutative Swap          | 4j + y + 20y + 17x + 9j + 9p
    STEP[27]: Distributive Factoring    | 4j + (1 + 20) * y + 17x + 9j + 9p
    STEP[28]: Constant Arithmetic       | 4j + 21y + 17x + 9j + 9p
    STEP[29]: Commutative Swap          | 21y + 4j + 17x + 9j + 9p
    STEP[30]: Commutative Swap          | 21y + 4j + 17x + 9p + 9j
    STEP[31]: Commutative Swap          | 21y + 4j + 9p + 17x + 9j
    STEP[32]: Commutative Swap          | 21y + 4j + 17x + 9p + 9j
    STEP[33]: Commutative Swap          | 4j + 21y + 17x + 9p + 9j
    STEP[34]: Commutative Swap          | 4j + 21y + 9p + 17x + 9j
    STEP[35]: Commutative Swap          | 4j + 21y + 9p + 9j + 17x
    STEP[36]: Commutative Swap          | 21y + 4j + 9p + 9j + 17x
    STEP[37]: Commutative Swap          | 21y + 4j + 9p + 17x + 9j
    STEP[38]: Commutative Swap          | 21y + 9p + 4j + 17x + 9j
    STEP[39]: Commutative Swap          | 21y + 9p + 17x + 4j + 9j
    STEP[40]: Distributive Factoring    | 21y + 9p + 17x + (4 + 9) * j
    STEP[41]: Constant Arithmetic       | 21y + 9p + 17x + 13j
    FINAL: 21y + 9p + 17x + 13j ---- WIN





    '21y + 9p + 17x + 13j'



## Reader's Challenge

The example problems we simplified don't include subtraction operators because the `commutative property` that moves nodes around cannot be applied to subtraction. Luckily for us, we can restate a subtraction as the addition of a negation. This allows us to commute the subtraction around while maintaining the value of the original expression.

Mathy core includes a built-in rule [Restate Subtraction](https://core.mathy.ai/api/rules/restate_subtraction/) rule that you can add to our `simplify_polynomial` function. Try adding this rule to the `simplify_polynomial` so that the following cell executes and simplifies each example successfully.

<details>
<summary>Click here for a hint</summary>
<em>It's not as simple as inserting the rule in the list.</em>
</details>
<details>
<summary>Click here for another hint</summary>
<em>You can remove the commutative rule from the ordered list of actions, and create another list with commutative and restate, then select randomly from the second list when none of the ordered rules are applicable.</em>
</details>


```python
simplify_polynomial("4x - 3x")
simplify_polynomial("4x - 3y + 3x")
```

## Conclusion

If you've made it this far, congratulations! Hopefully you have a basic grasp of how you can work with mathy_core to construct applications that solve specific types of math problems, while demonstrating their work step-by-step in an interpretable way.

While this example may be brittle, mathy is capable of much more if you combine it a formal environment in which to run simulations like the one we wrote here. That's where there library [mathy_envs](https://envs.mathy.ai) come into play, and we enter the exciting world of planning and learning algorithms! 

But we'll talk about that another time. Until then, happy hacking friends! 😎
