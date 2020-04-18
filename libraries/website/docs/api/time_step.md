# mathy.time_step
TimeStep representing a step in the environment.

This file is a mostly direct copy of the implementation from the
[tf_agents](https://github.com/tensorflow/agents) library but has
the dependency on tensorflow removed along with advanced shape
features.

Mathy doesn't use these features and the overhead of loading tensorflow
to pass environment states around is not great for things like CLI start
times.

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
