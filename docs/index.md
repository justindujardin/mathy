
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
<a href="https://pypi.org/project/mathtastic" target="_blank">
    <img src="https://badge.fury.io/py/mathtastic.svg" alt="Package version">
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

* **Computer Algebra System**: Parse text strings into expression trees for manipulation and evaluation.
* **Tree Transformation Rules**: Math trees can be transformed by user-defined rules that do not change the value.
* **RL Environments**: A number of [Reinforcement Learning](/ml/reinforcement_learning) environments are provided for training agents that manipulate math trees.

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

__Input__:`1k + 210r + 7z + 11k + 10z`

Step                      | Text      
--------                  |-------------
input                     | 1k + 210r + 7z + 11k + 10z
commutative swap          | 11k + __(1k + 210r + 7z)__ + 10z
distributive factoring    | 11k + (1k + 210r) + __(7 + 10) * z__
distributive factoring    | __(11 + 1) * k__ + 210r + (7 + 10) * z
constant arithmetic       | (11 + 1) * k + 210r + __17z__
constant arithmetic       | __12k__ + 210r + 17z
__solution__              | __12k + 210r + 17z__


### Binomial Multiplication

Mathy can perform the FOIL method: 

__Input__:`(k^4 + 7)(4 + h^2)`

Step                      | Text      
--------                  |-------------
initial                   | (k^4 + 7)(4 + h^2)
distributive multiply     | (4 + h^2) * k^4 + (4 + h^2) * 7
distributive multiply     | 4k^4 + k^4 * h^2 + (4 + h^2) * 7
commutative swap          | 4k^4 + k^4 * h^2 + 7 * (4 + h^2)
distributive multiply     | 4k^4 + k^4 * h^2 + (7 * 4 + 7h^2)
constant arithmetic       | 4k^4 + k^4 * h^2 + (28 + 7h^2)
__solution__              | __4k^4 + k^4 * h^2 + 28 + 7h^2__

### Complex Terms

Mathy can simplify complex terms:

__Input__: `4a^4 * 5a^4 * 2b^4`

Step                      | Text      
--------                  |-------------
initial                   | 4a^4 * 5a^4 * 2b^4
constant arithmetic       | __20a^4__ * a^4 * 2b^4
variable multiplication   | __20 * a^(4 + 4)__ * 2b^4
constant arithmetic       | 20 * __a^8__ * 2b^4
commutative swap          | __(a^8 * 2b^4)__ * 20
commutative swap          | (__2b^4 * a^8__) * 20
commutative swap          | 20 * __2b^4 * a^8__
constant arithmetic       | __40b^4__ * a^8
__solution__              | __40b^4 * a^8__
