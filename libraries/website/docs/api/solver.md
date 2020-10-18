# mathy.solver
Use Fractal Monte Carlo search in order to solve mathy problems without a
trained neural network.
## FragileEnvironment <kbd>class</kbd>
```python
FragileEnvironment(
    self, 
    name: str, 
    environment: str = 'poly', 
    difficulty: str = 'normal', 
    problem: str = None, 
    max_steps: int = 64, 
    kwargs, 
)
```
Fragile Environment for solving Mathy problems.
## FragileMathyEnv <kbd>class</kbd>
```python
FragileMathyEnv(
    self, 
    name: str, 
    environment: str = 'poly', 
    difficulty: str = 'easy', 
    problem: str = None, 
    max_steps: int = 64, 
    kwargs, 
)
```
The DiscreteEnv acts as an interface with `plangym` discrete actions.

It can interact with any environment that accepts discrete actions and     follows the interface of `plangym`.

### make_transitions <kbd>method</kbd>
```python
FragileMathyEnv.make_transitions(
    self, 
    states: numpy.ndarray, 
    actions: numpy.ndarray, 
    dt: Union[numpy.ndarray, int], 
) -> Dict[str, numpy.ndarray]
```

Step the underlying :class:`plangym.Environment` using the ``step_batch``         method of the ``plangym`` interface.

