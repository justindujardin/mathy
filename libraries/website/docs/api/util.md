# mathy.util

## discount
```python
discount(values:List[float], gamma=0.99) -> List[float]
```
Discount a list of floating point values.

__Arguments__

- __r (List[float])__: the list of floating point values to discount
- __gamma (float)__: a value between 0 and 0.99 to use when discounting the inputs

__Returns__

`(List[float])`: a list of the same size as the input with discounted values

## pad_array
```python
pad_array(in_list:List[Any], max_length:int, value:Any=0) -> List[Any]
```
Pad a list to the given size with the given padding value.

__Arguments:__

in_list (List[Any]): List of values to pad to the given length
max_length (int): The desired length of the array
value (Any): a value to insert in order to pad the array to max length

__Returns__

`(List[Any])`: An array padded to `max_length` size

