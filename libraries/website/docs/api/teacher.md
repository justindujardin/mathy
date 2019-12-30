# mathy.teacher

## Teacher
```python
Teacher(
    self,
    topic_names: List[str],
    num_students: int = 1,
    difficulty: Union[str, NoneType] = None,
    eval_window: int = 50,
    win_threshold: float = 0.95,
    lose_threshold: float = 0.34,
)
```

### get_env
```python
Teacher.get_env(self, student_index:int, iteration:int) -> str
```
Get the current environment a student should be using.

__Arguments__

- __student_index (int)__: The index of the student in `self.students` array.
- __iteration (int)__: The current iteration (usually an episode).

__Returns__

`(str)`: The name of a mathy environment to use.

### next_difficulty
```python
Teacher.next_difficulty(self, difficulty:str) -> str
```
Return the previous difficulty level given an input difficulty.

__Arguments__

- __difficulty (str)__: The difficulty to increase

__Returns__

`(str)`: The difficulty level after the input, if any.

### previous_difficulty
```python
Teacher.previous_difficulty(self, difficulty:str) -> str
```
Return the previous difficulty level given an input difficulty.

__Arguments__

- __difficulty (str)__: The difficulty to decrease

__Returns__

`(str)`: The difficulty level before the input, if any.

