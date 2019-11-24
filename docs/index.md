
![Mathy](img/math_orange.svg)

Mathy wants to be your free math tutor. It uses machine learning to solve math problems step-by-step. It is inspired by observing that the most useful feedback when trying to learn a new math concept is watching someone work through specific examples, explaining the thought process at each step. In this way it is able to work through as many example problems as a student needs to see to help understand a concept. This is only one type of problem, but Mathy has an extensible architecture making it easy to build new types of problems.

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
