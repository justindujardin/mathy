<p align="center">
  <a href="https://mathy.ai"><img src="img/mathy_logo.png" alt="Mathy.ai"></a>
</p>
<p align="center">
    <em>Computer Algebra Software library and Reinforcement Learning environment platform, with agents that solve math problems step-by-step</em>
</p>
<p align="center">
<a href="https://travis-ci.org/justindujardin/mathy" target="_blank">
    <img src="https://travis-ci.org/justindujardin/mathy.svg?branch=master" alt="Build Status">
</a>
<a href="https://codecov.io/gh/justindujardin/mathy" target="_blank">
    <img src="https://codecov.io/gh/justindujardin/mathy/branch/master/graph/badge.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/mathy" target="_blank">
    <img src="https://badge.fury.io/py/mathy.svg" alt="Package version">
</a>
<a href="https://gitter.im/justindujardin/mathy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge" target="_blank">
    <img src="https://badges.gitter.im/justindujardin/mathy.svg" alt="Join the chat at https://gitter.im/justindujardin/mathy">
</a>
</p>

---

**Documentation**: <a href="https://mathy.ai" target="_blank">https://mathy.ai</a>

**Source Code**: <a href="https://github.com/justindujardin/mathy" target="_blank">https://github.com/justindujardin/mathy</a>

---

Mathy wants to be your free math tutor. It uses machine learning to solve math problems step-by-step.

The key features are:

- **Computer Algebra System**: Parse text strings into expression trees for manipulation and evaluation.
- **Tree Transformation Rules**: Math trees can be transformed by user-defined rules that do not change the value.
- **RL Environments**: A number of [Reinforcement Learning](/ml/reinforcement_learning) environments are provided for training agents that manipulate math trees.

## Requirements

- Python 3.6+
- Tensorflow 2.0+

## Installation

```
$ pip install mathy
```

## Getting Started

```
$ mathy guess "2x + 4 + 3x * 6"
```

## Examples

### Combine Like Terms

Mathy can simplify expressions by reordering and combining like terms:

**Input**:`1k + 210r + 7z + 11k + 10z`

| Step                   | Text                                     |
| ---------------------- | ---------------------------------------- |
| input                  | 1k + 210r + 7z + 11k + 10z               |
| commutative swap       | 11k + **(1k + 210r + 7z)** + 10z         |
| distributive factoring | 11k + (1k + 210r) + **(7 + 10) \* z**    |
| distributive factoring | **(11 + 1) \* k** + 210r + (7 + 10) \* z |
| constant arithmetic    | (11 + 1) \* k + 210r + **17z**           |
| constant arithmetic    | **12k** + 210r + 17z                     |
| solution               | **12k + 210r + 17z**                     |

### Binomial Multiplication

Mathy can perform the FOIL method:

**Input**:`(k^4 + 7)(4 + h^2)`

| Step                  | Text                              |
| --------------------- | --------------------------------- |
| initial               | (k^4 + 7)(4 + h^2)                |
| distributive multiply | (4 + h^2) _ k^4 + (4 + h^2) _ 7   |
| distributive multiply | 4k^4 + k^4 _ h^2 + (4 + h^2) _ 7  |
| commutative swap      | 4k^4 + k^4 _ h^2 + 7 _ (4 + h^2)  |
| distributive multiply | 4k^4 + k^4 _ h^2 + (7 _ 4 + 7h^2) |
| constant arithmetic   | 4k^4 + k^4 \* h^2 + (28 + 7h^2)   |
| solution              | **4k^4 + k^4 \* h^2 + 28 + 7h^2** |

### Complex Terms

Mathy can simplify complex terms:

**Input**: `4a^4 * 5a^4 * 2b^4`

| Step                    | Text                        |
| ----------------------- | --------------------------- |
| initial                 | 4a^4 _ 5a^4 _ 2b^4          |
| constant arithmetic     | **20a^4** _ a^4 _ 2b^4      |
| variable multiplication | **20 \* a^(4 + 4)** \* 2b^4 |
| constant arithmetic     | 20 _ **a^8** _ 2b^4         |
| commutative swap        | **(a^8 \* 2b^4)** \* 20     |
| commutative swap        | (**2b^4 \* a^8**) \* 20     |
| commutative swap        | 20 _ __2b^4 _ a^8__     |
| constant arithmetic     | **40b^4** \* a^8            |
| solution                | **40b^4 \* a^8**            |
