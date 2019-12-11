# mathy.time_step
TimeStep representing a step in the environment.
## StepType
```python
StepType(self, /, *args, **kwargs)
```
Defines the status of a `TimeStep` within a sequence.
## termination
```python
termination(observation, reward)
```
Returns a `TimeStep` with `step_type` set to `StepType.LAST`.
## transition
```python
transition(observation, reward, discount=1.0)
```
Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.
## truncation
```python
truncation(observation, reward, discount=1.0)
```
Returns a `TimeStep` with `step_type` set to `StepType.LAST`.
