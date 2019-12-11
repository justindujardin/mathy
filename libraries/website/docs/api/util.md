# mathy.util

## discount
```python
discount(r, gamma=0.99)
```
Discount a list of float rewards to encourage rapid convergance.
r: input array of floats
gamma: a float value between 0 and 0.99
## EnvRewards
```python
EnvRewards(self, /, *args, **kwargs)
```
Game reward constant values
### HELPFUL_MOVE
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### INVALID_ACTION
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### LOSE
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### PREVIOUS_LOCATION
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### TIMESTEP
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### UNHELPFUL_MOVE
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### WIN
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
## normalize_rewards
```python
normalize_rewards(r)
```
Normalize a set of rewards to values between -1 and 1
## pad_array
```python
pad_array(A, max_length, value=0, backwards=False, cleanup=False)
```
Pad a list to the given size with the given padding value

If backwards=True the input will be reversed after padding, and
the output will be reversed after padding, to correctly pad for
LSTMs, e.g. "4x+2----" padded backwards would be "----2+x4"

