<p align="center">
  <a href="/"><img mathy-logo src="/img/mathy_logo.png" alt="Mathy.ai"></a>
</p>
<p align="center">
    <em>A modern computer algebra system and reinforcement learning environments platform for interpretable symbolic mathematics.</em>
</p>
<p align="center">
<a href="https://github.com/justindujardin/mathy/actions">
    <img src="https://github.com/justindujardin/mathy/workflows/Build/badge.svg" />
</a>
<a href="https://codecov.io/gh/justindujardin/mathy">
    <img src="https://codecov.io/gh/justindujardin/mathy/branch/master/graph/badge.svg?token=CqPEOdEMJX" />
</a>
<a href="https://pypi.org/project/mathy" target="_blank">
    <img src="https://badge.fury.io/py/mathy.svg" alt="Package version">
</a>
<a href="https://gitter.im/justindujardin/mathy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge" target="_blank">
    <img src="https://badges.gitter.im/justindujardin/mathy.svg" alt="Join the chat at https://gitter.im/justindujardin/mathy">
</a>
</p>
<div align="center" data-termynal-container>
    <div id="termynal" data-termynal="" data-ty-typedelay="40" data-ty-lineDelay="1000">
        <span data-ty="input">pip install mathy mathy_alpha_sm</span>
        <span data-ty="progress"></span>
        <span class="u-hide-sm" data-ty-lineDelay="0" data-ty="">Successfully installed mathy</span>
        <span data-ty-lineDelay="0" class="u-hide-sm" data-ty=""></span>
        <span data-ty="input">mathy simplify "2x + 1y^3 + 7b + 4x"</span>
        <span data-ty="" data-ty-text="initial                   | 2x + 1y^3 + 7b + 4x"></span>
        <span data-ty="" data-ty-text="associative group         | 2x + (1y^3 + 7b) + 4x"></span>
        <span data-ty="" data-ty-text="associative group         | 2x + (1y^3 + 7b + 4x)"></span>
        <span data-ty="" data-ty-text="commutative swap          | 1y^3 + 7b + 4x + 2x"></span>
        <span data-ty="" data-ty-text="associative group         | 1y^3 + 7b + (4x + 2x)"></span>
        <span data-ty="" data-ty-text="distributive factoring    | 1y^3 + 7b + (4 + 2) * x"></span>
        <span data-ty="" data-ty-text="constant arithmetic       | 1y^3 + 7b + 6x"></span>
        <span data-ty-lineDelay="0" class="u-hide-sm" data-ty=""></span>
        <span data-ty="" data-ty-text='"2x + 1y^3 + 7b + 4x" = "1y^3 + 7b + 6x"'></span>
    </div>
</div>

---

**Documentation**: <a href="https://mathy.ai" target="_blank">https://mathy.ai</a>

**Source Code**: <a href="https://github.com/justindujardin/mathy" target="_blank">https://github.com/justindujardin/mathy</a>

---

Mathy wants to be your free and open source math tutor. It uses machine learning to manipulate math problems step-by-step.

## Features

- **[Computer Algebra System](/cas/overview)**: Parse text into expression trees for manipulation and evaluation. Transform trees with user-defined rules that do not change the value of the expression.
- **[Reinforcement learning](/ml/reinforcement_learning)**: Train agents with machine learning in many environments with hyperparameters for controlling environment difficulties.
- **[Custom Environments](/envs/overview)** Extend built-in environments or author your own. Provide custom logic and values for custom actions, problems, timestep rewards, episode rewards, and win-conditions.
- **[Visualize Expressions](/api/core/layout)**: Gain a deeper understanding of problem structures and rule transformations by visualizing binary trees in a compact layout with no branch overlaps.
- **[Compute Friendly](/ml/a3c)**: Maybe we don't have to burn down the world with GPU compute all the time? Text-based environments can be small enough to train on a CPU while still having real-world value.
- **[Free and Open Source](/license)**: Mathy is and will always be free, because educational tools are too important to our world to be gated by money.

## Requirements

- Python 3.6+
- Tensorflow 2.0+

## Installation

```bash
$ pip install mathy mathy_alpha_sm
```

## Try It

Let's start by simplifying a polynomial problem using the CLI:

```bash
$ mathy simplify "2x + 4 + 3x * 6"
```

This uses the pretrained `mathy_alpha_sm` model that we installed above.

The model is used to determine which intermediate steps to take in order to get to the desired solution.

The output will vary based on the model, but it might look like this:

<div align="center" data-termynal-container>
    <div id="termynal-two" data-termynal="" data-ty-typedelay="40" data-ty-lineDelay="1000">
        <span data-ty="input">mathy simplify "2x + 4 + 3x * 6"</span>
        <span data-ty="" data-ty-text="initial                   | 2x + 4 + 3x * 6"></span>
        <span data-ty="" data-ty-text="constant arithmetic       | 2x + 4 + 18x"></span>
        <span data-ty="" data-ty-text="commutative swap          | 4 + 2x + 18x"></span>
        <span data-ty="" data-ty-text="commutative swap          | 2x + 4 + 18x"></span>
        <span data-ty="" data-ty-text="commutative swap          | 18x + (2x + 4)"></span>
        <span data-ty="" data-ty-text="distributive factoring    | (18 + 2) * x + 4"></span>
        <span data-ty="" data-ty-text="constant arithmetic       | 20x + 4"></span>
        <span data-ty-lineDelay="0" class="u-hide-sm" data-ty=""></span>
        <span data-ty="" data-ty-text='"2x + 4 + 3x * 6" = "20x + 4"'></span>
    </div>
</div>

## Code It

Above we simplified a polynomial problem using the CLI, but what if the output steps had failed to find a solution?

Perhaps we put a [subtraction](/api/core/expressions/#subtractexpression) between two like terms.

Recall that we can't move subtraction terms around with the [commutative property](/rules/commutative_property), so how can Mathy solve this problem?

We can write custom code for Mathy in order to add features or correct issues.

!!! tip "The power of Open Source"

    By contributing improvements to Mathy, we help ourselves better understand Math and Programming.

    We also create examples for others around the world that are trying to get help with Math or learn Programming!

In order to combine these terms, we need to convert the subtraction into an addition.

Remember that a subtraction (e.g. `4x - 2x`) can be restated as a "plus negative" (`4x + -2x`) in order to make it [commutable](/rules/commutative_property).

### Create a Rule

Mathy uses the available set of **[rules](/rules/overview)** (sometimes referred to as **actions**) when transforming a problem.

To create a custom rule, we extend the [BaseRule](/api/core/rule/#baserule) class and define two main functions:

- `can_apply_to` determines if a rule can be applied to an expression node.
- `apply_to` applies the rule to a node and returns an [expression change](/api/core/rule/#expressionchange) object.

Let's define a rule that looks for a [subtraction](/api/core/expressions/#subtractionexpression) and converts it to a "plus negation" node:

```Python
{!./snippets/create_a_rule.py!}
```

### Use it during training

Now that we've created a custom rule, we need Mathy to use it while training.

We do this with custom environment arguments when using the [A3C Agent](/ml/a3c).

All together it looks like:

```python
{!./snippets/use_a_custom_rule.py!}
```

## Contributors

Mathy wouldn't be possible without the wonderful contributions of the following people:

<div class="contributors-wrapper">
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a target="_blank" href="https://www.justindujardin.com/"><img src="https://avatars0.githubusercontent.com/u/101493?v=4" width="100px;" alt=""/><br /><sub><b>Justin DuJardin</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
</div>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

```

```
