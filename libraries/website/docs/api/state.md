# mathy.state

## MathyAgentState
```python
MathyAgentState(
    self,
    moves_remaining,
    problem,
    problem_type,
    reward = 0.0,
    history = None,
    focus_index = 0,
    last_action = None,
)
```
The state related to an agent for a given environment state
## MathyEnvState
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

### get_out_state
```python
MathyEnvState.get_out_state(
    self,
    problem: str,
    action: int,
    focus_index: int,
    moves_remaining: int,
) -> 'MathyEnvState'
```
Get the next environment state based on the current one with updated
history and agent information based on an action being taken.
### get_problem_hash
```python
MathyEnvState.get_problem_hash(self) -> List[int]
```
Return a two element array with hashed values for the current environment
namespace string.

__Example__


- `mycorp.envs.solve_impossible_problems` -> `[12375561, -2838517]`


### to_start_observation
```python
MathyEnvState.to_start_observation(self) -> mathy.state.MathyObservation
```
Generate an episode start MathyObservation
## MathyEnvStateStep
```python
MathyEnvStateStep(self, /, *args, **kwargs)
```
Capture summarized environment state for a previous timestep so the
agent can use context from its history when making new predictions.
### action
the action taken
### focus
the index of the node that is acted on
### raw
the input text at the timestep
## MathyObservation
```python
MathyObservation(self, /, *args, **kwargs)
```
A featurized observation from an environment state.
### mask
0/1 mask where 0 indicates an invalid action shape=[n,]
### nodes
tree node types in the current environment state shape=[n,]
### time
float value between 0.0-1.0 for how much time has passed shape=[1,]
### type
two column hash of problem environment type shape=[2,]
### values
tree node value sequences, with non number indices set to 0.0 shape=[n,]
## MathyWindowObservation
```python
MathyWindowObservation(self, /, *args, **kwargs)
```
A featurized observation from an n-step sequence of environment states.
### mask
n-step list of node sequence masks `shape=[n, max(len(s))]`
### nodes
n-step list of node sequences `shape=[n, max(len(s))]`
### time
float value between 0.0-1.0 for how much time has passed `shape=[n, 1]`
### type
n-step problem type hash `shape=[n, 2]`
### values
n-step list of value sequences, with non number indices set to 0.0 `shape=[n, max(len(s))]`
## ObservationFeatureIndices
```python
ObservationFeatureIndices(self, /, *args, **kwargs)
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
## observations_to_window
```python
observations_to_window(
    observations: List[mathy.state.MathyObservation],
    total_length: int = None,
) -> mathy.state.MathyWindowObservation
```
Combine a sequence of observations into an observation window
