<p align="center">
  <a href="/"><img mathy-logo src="/img/mathy_logo.png" alt="Mathy.ai"></a>
</p>
<p align="center">
    <em>A modern computer algebra system and reinforcement learning platform for interpretable symbolic mathematics.</em>
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
        <span data-ty="input">pip install mathy</span>
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
- **[Reinforcement learning](/ml/reinforcement_learning)**: Train agents with machine learning, in many environments, with variable hyperparameters for controlling problem difficulties.
- **[Custom Environments](/envs/overview)** Extend built-in environments or author your own. Provide custom logic and values for custom actions, problems, timestep rewards, episode rewards, and win-conditions.
- **[Visualize Expressions](/api/core/layout)**: Gain a deeper understanding of problem structures and rule transformations by visualizing binary trees in a compact layout with no branch overlaps.
  <!-- - **Portable Implementation**: Originally a C# project, then CoffeeScript, then Typescript, and now Python. Mathy's simple design ports easily to many languages. -->
  <!-- - **Language-agnostic tests**: JSON-based assertions verify that systems execute consistently across platform implementations. -->

## Example

Consider an example where you want to know the solution to a binomial distribution problem:

### Input

`(k^4 + 7)(4 + h^2)`

`mathy:(k^4 + 7)(4 + h^2)`

### Steps

| Step                  | Text                                |
| --------------------- | ----------------------------------- |
| initial               | (k^4 + 7)(4 + h^2)                  |
| distributive multiply | (4 + h^2) \* k^4 + (4 + h^2) \* 7   |
| distributive multiply | 4k^4 + k^4 \* h^2 + (4 + h^2) \* 7  |
| commutative swap      | 4k^4 + k^4 \* h^2 + 7 \* (4 + h^2)  |
| distributive multiply | 4k^4 + k^4 \* h^2 + (7 \* 4 + 7h^2) |
| constant arithmetic   | 4k^4 + k^4 \* h^2 + (28 + 7h^2)     |
| solution              | 4k^4 + k^4 \* h^2 + 28 + 7h^2       |

### Solution

`4k^4 + k^4 * h^2 + 28 + 7h^2`

`mathy:4k^4 + k^4 * h^2 + 28 + 7h^2`

## Requirements

- Python 3.6+
- Tensorflow 2.0+

## Installation

```
$ pip install mathy
```

## Getting Started

Mathy can be used as a command-line tool to simplify basic polynomial problems.

```
$ mathy simplify "2x + 4 + 3x * 6"
```
