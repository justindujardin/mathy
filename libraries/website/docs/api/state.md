# mathy.state

## MathyAgentState <kbd>class</kbd>
```python
MathyAgentState(
    self, 
    moves_remaining, 
    problem, 
    problem_type, 
    reward = 0.0, 
    history = None, 
)
```
The state related to an agent for a given environment state
## MathyEnvState <kbd>class</kbd>
```python
MathyEnvState(
    self, 
    state: Optional[MathyEnvState] = None, 
    problem: str = None, 
    max_moves: int = 10, 
    num_rules: int = 0, 
    problem_type: str = 'mathy.unknown', 
)
```
Class for holding environment state and extracting features
to be passed to the policy/value neural network.

Mutating operations all return a copy of the environment adapter
with its own state.

This allocation strategy requires more memory but removes a class
of potential issues around unintentional sharing of data and mutation
by two different sources.

### from_np <kbd>classmethod</kbd>
```python
MathyEnvState.from_np(input_bytes: numpy.ndarray) -> 'MathyEnvState'
```
Convert a numpy object into a state object
### from_string <kbd>classmethod</kbd>
```python
MathyEnvState.from_string(input_string: str) -> 'MathyEnvState'
```
Convert a string representation of state into a state object
### get_out_state <kbd>method</kbd>
```python
MathyEnvState.get_out_state(
    self, 
    problem: str, 
    focus: int, 
    action: int, 
    moves_remaining: int, 
) -> 'MathyEnvState'
```
Get the next environment state based on the current one with updated
history and agent information based on an action being taken.
### get_problem_hash <kbd>method</kbd>
```python
MathyEnvState.get_problem_hash(self) -> List[int]
```
Return a two element array with hashed values for the current environment
namespace string.

__Example__


- `mycorp.envs.solve_impossible_problems` -> `[12375561, -2838517]`


### to_np <kbd>method</kbd>
```python
MathyEnvState.to_np(self) -> numpy.ndarray
```
Convert a state object into a numpy representation
### to_observation <kbd>method</kbd>
```python
MathyEnvState.to_observation(
    self, 
    move_mask: Optional[List[int]] = None, 
    hash_type: Optional[List[int]] = None, 
    parser: Optional[mathy_core.parser.ExpressionParser] = None, 
) -> mathy.state.MathyObservation
```
Convert a state into an observation
### to_start_observation <kbd>method</kbd>
```python
MathyEnvState.to_start_observation(self) -> mathy.state.MathyObservation
```
Generate an episode start MathyObservation
### to_string <kbd>method</kbd>
```python
MathyEnvState.to_string(self) -> str
```
Convert a state object into a string representation
## MathyEnvStateStep <kbd>class</kbd>
```python
MathyEnvStateStep(self, args, kwargs)
```
Capture summarized environment state for a previous timestep so the
agent can use context from its history when making new predictions.
### action
the action taken
### focus
the index of the node that is acted on
### raw
the input text at the timestep
## MathyObservation <kbd>class</kbd>
```python
MathyObservation(self, args, kwargs)
```
A featurized observation from an environment state.
### mask
0/1 mask where 0 indicates an invalid action shape=[n,]
### nodes
tree node types in the current environment state shape=[n,]
### time
float value between 0.0 and 1.0 indicating the time elapsed shape=[1,]
### type
two column hash of problem environment type shape=[2,]
### values
tree node value sequences, with non number indices set to 0.0 shape=[n,]
## MathyWindowObservation <kbd>class</kbd>
```python
MathyWindowObservation(self, args, kwargs)
```
A featurized observation from an n-step sequence of environment states.
### mask
n-step list of node sequence masks `shape=[n, max(len(s))]`
### nodes
n-step list of node sequences `shape=[n, max(len(s))]`
### time
n-step problem time values `shape=[n, 2]`
### type
n-step problem type hash `shape=[n, 2]`
### values
n-step list of value sequences, with non number indices set to 0.0 `shape=[n, max(len(s))]`
## ObservationFeatureIndices <kbd>class</kbd>
```python
ObservationFeatureIndices(self, args, kwargs)
```
An enumeration.
### mask
An enumeration.
### nodes
An enumeration.
### time
An enumeration.
### type
An enumeration.
### values
An enumeration.
## observations_to_window <kbd>function</kbd>
```python
observations_to_window(
    observations: List[mathy.state.MathyObservation], 
    pad_length: int, 
) -> mathy.state.MathyWindowObservation
```
Combine a sequence of observations into an observation window
