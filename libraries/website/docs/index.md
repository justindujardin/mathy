<p align="center">
  <a href="/"><img mathy-logo src="/img/mathy_logo.png" alt="Mathy.ai"></a>
</p>
<p align="center">
    <em>Solve math problems. Show your work. Contribute new solution types.</em>
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

## Features

- **[Computer Algebra System](/cas/overview)**: Parse text into expression trees for manipulation and evaluation. Transform trees with user-defined rules that do not change the value of the expression.
- **[Learning Environments](/envs/overview)** Use built-in environments or author your own. Provide custom logic and values for custom actions, problems, timestep rewards, episode rewards, and win-conditions.
- **[Visualize Expressions](/api/core/layout)**: Gain a deeper understanding of problem structures and rule transformations by visualizing binary trees in a compact layout with no branch overlaps.
- **[Free and Open Source](/license)**: Mathy is free because educational tools are important and should be accessible to everyone.
- **[Python with Type Hints](https://fastapi.tiangolo.com/python-types/){target=\_blank}**: typing hints are used everywhere in Mathy to help provide rich autocompletion and linting in modern IDEs.

## Requirements

- Python 3.6+

## Installation

```bash
$ pip install mathy
```

## Try It

Let's start by simplifying a polynomial problem using the CLI:

### Simplify a Polynomial

```bash
$ mathy simplify "2x + 4 + 3x * 6"
```

Mathy uses a swam planning algorithm to determine which intermediate steps to take to get to the desired solution.

The output will vary based, but it might look like this:

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

### Generate Input Problems

Mathy can generate lists of randomized problems. Rather than forcing users to create solutions, Mathy uses environment-specific functions to determine when a problem is solved. In this way, users don't need to know the answer to a question that they generate.

```bash
$ mathy problems poly
```

## Contributors

Mathy wouldn't be possible without the contributions of the following people:

<div class="contributors-wrapper">
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a target="_blank" href="https://www.justindujardin.com/"><img src="https://avatars0.githubusercontent.com/u/101493?v=4" width="100px;" alt=""/><br /><sub><b>Justin DuJardin</b></sub></a></td>
    <td align="center"><a target="_blank" href="https://twitter.com/Miau_DB"><img src="https://avatars3.githubusercontent.com/u/7149899?v=4" width="100px;" alt=""/><br /><sub><b>Guillem Duran Ballester</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
</div>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
