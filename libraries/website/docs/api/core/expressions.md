# mathy.core.expressions

## AbsExpression
```python
AbsExpression(self, child=None, operatorOnLeft=True)
```
Evaluates the absolute value of an expression.
### differentiate
```python
AbsExpression.differentiate(self, by_variable)
```
```
.       f(x)   = abs( g(x) )
.    d( f(x) ) = sgn( g(x) ) * d( g(x) )
```
## AddExpression
```python
AddExpression(self, left=None, right=None)
```
Add one and two
### differentiate
```python
AddExpression.differentiate(self, by_variable)
```
```
.           f(x) = g(x) + h(x)
.      d( f(x) ) = d( g(x) ) + d( h(x) )
.          f'(x) = g'(x) + h'(x)
```
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

### self_parens
```python
BinaryExpression.self_parens(self) -> bool
```
Return a boolean indicating whether this node should render itself with
a set of enclosing parnetheses or not. This is used when serializing an
expression, to ensure the tree maintains the proper order of operations.
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
### differentiate
```python
DivideExpression.differentiate(self, by_variable)
```
```
.      f(x) = g(x)/h(x)
.     f'(x) = ( g'(x)*h(x) - g(x)*h'(x) ) / ( h(x)^2 )
```
## EqualExpression
```python
EqualExpression(self, left=None, right=None)
```
Evaluate equality of two expressions
### operate
```python
EqualExpression.operate(self, one:mathy.core.expressions.BinaryExpression, two:mathy.core.expressions.BinaryExpression)
```
This is where assignment of context variables might make sense. But context
is not present in the expression's `operate` method.

!!! warning

    TODO: Investigate this thoroughly.

## FunctionExpression
```python
FunctionExpression(self, child=None, operatorOnLeft=True)
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
format.  See `MathExpression.to_math_ml_fragment`
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
subtree node. See `MathExpression.clone_from_root` for more details.
### clone_from_root
```python
MathExpression.clone_from_root(self, node=None) -> 'MathExpression'
```
Clone this node including the entire parent hierarchy that it has. This
is useful when you want to clone a subtree and still maintain the overall
hierarchy.

Params:

    - `node` The node to clone.

Returns: The cloned `MathExpression` node.

### color
Color to use for this node when rendering it as changed with `.terminal_text`
### count_nodes
```python
MathExpression.count_nodes(self)
```
Return the number of nodes in this expression
### differentiate
```python
MathExpression.differentiate(self, by_variable)
```
Differentiate the expression by a given variable
### evaluate
```python
MathExpression.evaluate(self, context=None)
```
Evaluate the expression, resolving all variables to constant values
### find_id
```python
MathExpression.find_id(self, id)
```
Find an expression by its unique ID.

Returns: The found `MathExpression` or `None`

### find_type
```python
MathExpression.find_type(self, instanceType)
```
Find an expression in this tree by type.

- instanceType: The type to check for instances of

Returns the found `MathExpression` objects of the given type.

### make_ml_tag
```python
MathExpression.make_ml_tag(self, tag:str, content, classes=[]) -> str
```
Make a MathML tag for the given content while respecting the node's given
classes.

Params:

    - `tag` The ML tag name to create.
    - `content` The ML content to place inside of the tag.
    - `classes` An array of classes to attach to this tag.

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
### differentiate
```python
MultiplyExpression.differentiate(self, by_variable)
```
```
.         f(x) = g(x)*h(x)
.        f'(x) = g(x)*h'(x) + g'(x)*h(x)
```
## NegateExpression
```python
NegateExpression(self, child=None, operatorOnLeft=True)
```
Negate an expression, e.g. `4` becomes `-4`
### differentiate
```python
NegateExpression.differentiate(self, by_variable)
```

```
.        f(x) = -g(x)
.    d( f(x) ) = -( d( g(x) ) )
```

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
### differentiate
```python
PowerExpression.differentiate(self, by_variable)
```

!!! warn Unimplemented

    This needs to be implemented

## SgnExpression
```python
SgnExpression(self, child=None, operatorOnLeft=True)
```

### differentiate
```python
SgnExpression.differentiate(self, by_variable)
```
```
.         f(x) = sgn( g(x) )
.    d( f(x) ) = 0
```

Note: in general sgn'(x) = 2δ(x) where δ(x) is the Dirac delta function.
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
### differentiate
```python
SubtractExpression.differentiate(self, by_variable)
```
```
.           f(x) = g(x) - h(x)
.      d( f(x) ) = d( g(x) ) - d( h(x) )
.          f'(x) = g'(x) - h'(x)
```
## UnaryExpression
```python
UnaryExpression(self, child=None, operatorOnLeft=True)
```
An expression that operates on one sub-expression
## VariableExpression
```python
VariableExpression(self, identifier=None)
```

### differentiate
```python
VariableExpression.differentiate(self, by_variable)
```

Differentiating by this variable yields 1

```
.         f(x) = x
.    d( f(x) ) = 1 * d( x )
.       d( x ) = 1
.        f'(x) = 1
```

Differentiating by any other variable yields 0

```
.         f(x) = c
.    d( f(x) ) = c * d( c )
.       d( c ) = 0
.        f'(x) = 0
```

