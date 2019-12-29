# mathy.teacher

## Teacher
```python
Teacher(self, topic_names:List[str], num_students:int=1, difficulty:Union[str, NoneType]=None, eval_window:int=50, win_threshold:float=0.95, lose_threshold:float=0.34)
```

### get_directed_topic
```python
Teacher.get_directed_topic(self, eval_win_ratio:Union[float, NoneType]=None) -> str
```
After each evaluation, student zero gets a new topic.
