# mathy.core.expressions

## AbsExpression
```python
AbsExpression(self, child=None, child_on_left=True)
```
Evaluates the absolute value of an expression.
## AddExpression
```python
AddExpression(self, left=None, right=None)
```
Add one and two
## BinaryExpression
```python
BinaryExpression(self, left=None, right=None)
```
An expression that operates on two sub-expressions
### get_priority
```python
BinaryExpression.get_priority(self)
```
Return a number representing the order of operations priority
of this node.  This can be used to check if a node is `locked`
with respect to another node, i.e. the other node must be resolved
first during evaluation because of it's priority.

### to_math_ml_fragment
```python
BinaryExpression.to_math_ml_fragment(self)
```
Render this node as a MathML element fragment
## ConstantExpression
```python
ConstantExpression(self, value=None)
```
A Constant value node, where the value is accessible as `node.value`
## DivideExpression
```python
DivideExpression(self, left=None, right=None)
```
Divide one by two
## EqualExpression
```python
EqualExpression(self, left=None, right=None)
```
Evaluate equality of two expressions
### operate
```python
EqualExpression.operate(
    self,
    one: mathy.core.expressions.BinaryExpression,
    two: mathy.core.expressions.BinaryExpression,
)
```
This is where assignment of context variables might make sense. But context
is not present in the expression's `operate` method.

!!! warning

    TODO: Investigate this thoroughly.

## FunctionExpression
```python
FunctionExpression(self, child=None, child_on_left=True)
```
A Specialized UnaryExpression that is used for functions.  The function name in
text (used by the parser and tokenizer) is derived from the name() method on the
class.
## MathExpression
```python
MathExpression(self, id=None, left=None, right=None, parent=None)
```
Math tree node with helpers for manipulating expressions.

`mathy:x+y=z`

### add_class
```python
MathExpression.add_class(self, classes)
```
Associate a class name with an expression. This class name will be
attached to nodes when the expression is converted to a capable output
format.

See `MathExpression.to_math_ml_fragment`
### all_changed
```python
MathExpression.all_changed(self)
```
Mark this node and all of its children as changed
### clear_classes
```python
MathExpression.clear_classes(self)
```
Clear all the classes currently set on the nodes in this expression.
### clone
```python
MathExpression.clone(self) -> 'MathExpression'
```
A specialization of the clone method that can track and report a cloned
subtree node.

See `MathExpression.clone_from_root` for more details.
### clone_from_root
```python
MathExpression.clone_from_root(self, node=None) -> 'MathExpression'
```
Clone this node including the entire parent hierarchy that it has. This
is useful when you want to clone a subtree and still maintain the overall
hierarchy.

__Arguments__

- __node (MathExpression)__: The node to clone.

__Returns__

`(MathExpression)`: The cloned node.

### color
Color to use for this node when rendering it as changed with `.terminal_text`
### evaluate
```python
MathExpression.evaluate(self, context=None)
```
Evaluate the expression, resolving all variables to constant values
### find_id
```python
MathExpression.find_id(
    self,
    id: str,
) -> Union[_ForwardRef('MathExpression'), NoneType]
```
Find an expression by its unique ID.

Returns: The found `MathExpression` or `None`

### find_type
```python
MathExpression.find_type(
    self,
    instanceType: Type[_ForwardRef('MathExpression')],
) -> List[_ForwardRef('MathExpression')]
```
Find an expression in this tree by type.

- instanceType: The type to check for instances of

Returns the found `MathExpression` objects of the given type.

### make_ml_tag
```python
MathExpression.make_ml_tag(
    self,
    tag: str,
    content: str,
    classes: List[str] = [],
) -> str
```
Make a MathML tag for the given content while respecting the node's given
classes.

__Arguments__

- __tag (str)__: The ML tag name to create.
- __content (str)__: The ML content to place inside of the tag.
classes (List[str]) An array of classes to attach to this tag.

__Returns__

`(str)`: A MathML element with the given tag, content, and classes

### path_to_root
```python
MathExpression.path_to_root(self) -> str
```
Generate a namespaced path key to from the current node to the root.
This key can be used to identify a node inside of a tree.
### raw
raw text representation of the expression.
### set_changed
```python
MathExpression.set_changed(self)
```
Mark this node as having been changed by the application of a Rule
### terminal_text
Text output of this node that includes terminal color codes that
highlight which nodes have been changed in this tree as a result of
a transformation.
### to_list
```python
MathExpression.to_list(self, visit='preorder')
```
Convert this node hierarchy into a list.
### to_math_ml
```python
MathExpression.to_math_ml(self)
```
Convert this expression into a MathML container.
### to_math_ml_fragment
```python
MathExpression.to_math_ml_fragment(self)
```
Convert this single node into MathML.
### with_color
```python
MathExpression.with_color(self, text:str, style='bright') -> str
```
Render a string that is colored if something has changed
## MultiplyExpression
```python
MultiplyExpression(self, left=None, right=None)
```
Multiply one and two
## NegateExpression
```python
NegateExpression(self, child=None, child_on_left=True)
```
Negate an expression, e.g. `4` becomes `-4`
### to_math_ml_fragment
```python
NegateExpression.to_math_ml_fragment(self)
```
Convert this single node into MathML.
## PowerExpression
```python
PowerExpression(self, left=None, right=None)
```
Raise one to the power of two
## SgnExpression
```python
SgnExpression(self, child=None, child_on_left=True)
```

### operate
```python
SgnExpression.operate(self, value)
```
Determine the sign of an value.

Returns: -1 if negative, 1 if positive, 0 if 0
## SubtractExpression
```python
SubtractExpression(self, left=None, right=None)
```
Subtract one from two
## UnaryExpression
```python
UnaryExpression(self, child=None, child_on_left=True)
```
An expression that operates on one sub-expression
