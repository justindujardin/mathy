# mathy.core.expressions

## MathExpression
```python
MathExpression(self, id=None, left=None, right=None, parent=None)
```
Base math tree node with helpers for manipulating expressions.
### color
Color to use for this node when rendering it as changed with `.colored`
### raw
raw text representation of the expression.
### evaluate
```python
MathExpression.evaluate(self, context=None)
```
Evaluate the expression, resolving all variables to constant values
### set_changed
```python
MathExpression.set_changed(self)
```
Mark this node as having been changed by the application of a Rule
### all_changed
```python
MathExpression.all_changed(self)
```
Mark this node and all of its children as changed
### differentiate
```python
MathExpression.differentiate(self, by_variable)
```
Differentiate the expression by a given variable
### with_color
```python
MathExpression.with_color(self, text:str, style='bright') -> str
```
Render a string that is colored if the boolean input is True
### add_class
```python
MathExpression.add_class(self, classes)
```

Associate a class name with an expression.  This class name will be
tagged on nodes when the expression is converted to a capable output
format.  See `MathExpression.to_math_ml`

### set_weight
```python
MathExpression.set_weight(self, weight:float) -> 'MathExpression'
```
Associate a weight with this node. This is used when rendering attention
weights on expression nodes.
### count_nodes
```python
MathExpression.count_nodes(self)
```
Return the number of nodes in this expression
### toList
```python
MathExpression.toList(self, visit='postorder')
```
Convert this node hierarchy into a list.
### clear_classes
```python
MathExpression.clear_classes(self)
```
Clear all the classes currently set on the nodes in this expression.
### findByType
```python
MathExpression.findByType(self, instanceType)
```
Find an expression in this tree by type.

- instanceType: The type to check for instances of

Returns the found `MathExpression` objects of the given type.

### findById
```python
MathExpression.findById(self, id)
```
Find an expression by its unique ID.

Returns: The found `MathExpression` or `None`

### to_math_ml
```python
MathExpression.to_math_ml(self)
```
Convert this single node into MathML.
### to_math_ml_element
```python
MathExpression.to_math_ml_element(self)
```
Convert this expression into a MathML container.
### make_ml_tag
```python
MathExpression.make_ml_tag(self, tag, content, classes=[])
```

Make an ML tag for the given content, respecting the node's
given classes.
@param {String} tag The ML tag name to create.
@param {String} content The ML content to place inside of the tag.
@param {Array} classes An array of classes to attach to this tag.

### path_to_root
```python
MathExpression.path_to_root(self)
```

Generate a namespaced path key to from the current node to the root.
This key can be used to identify a node inside of a tree.

### clone_from_root
```python
MathExpression.clone_from_root(self, node=None)
```

Like the clone method, but also clones the parent hierarchy up to
the node root.  This is useful when you want to clone a subtree and still
maintain the overall hierarchy.
@param {MathExpression} [node=self] The node to clone.
@returns {MathExpression} The cloned node.

### clone
```python
MathExpression.clone(self)
```

A specialization of the clone method that can track and report a cloned subtree
node.  See {@link `clone_from_root`} for more details.

## UnaryExpression
```python
UnaryExpression(self, child=None, operatorOnLeft=True)
```
An expression that operates on one sub-expression
## NegateExpression
```python
NegateExpression(self, child=None, operatorOnLeft=True)
```
Negate an expression, e.g. `4` becomes `-4`
### to_math_ml
```python
NegateExpression.to_math_ml(self)
```
Convert this single node into MathML.
### differentiate
```python
NegateExpression.differentiate(self, by_variable)
```

<pre>
        f(x) = -g(x)
    d( f(x) ) = -( d( g(x) ) )
</pre>

## FunctionExpression
```python
FunctionExpression(self, child=None, operatorOnLeft=True)
```

A Specialized UnaryExpression that is used for functions.  The function name in
text (used by the parser and tokenizer) is derived from the name() method on
the class.

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

## EqualExpression
```python
EqualExpression(self, left=None, right=None)
```
Evaluate equality of two expressions
### operate
```python
EqualExpression.operate(self, one:mathy.core.expressions.BinaryExpression, two:mathy.core.expressions.BinaryExpression)
```

This is where assignment of context variables might make sense.  But context is not
present in the expression's `operate` method.  TODO: Investigate this thoroughly.

## AddExpression
```python
AddExpression(self, left=None, right=None)
```
Add one and two
## SubtractExpression
```python
SubtractExpression(self, left=None, right=None)
```
Subtract one from two
## MultiplyExpression
```python
MultiplyExpression(self, left=None, right=None)
```
Multiply one and two
## DivideExpression
```python
DivideExpression(self, left=None, right=None)
```
Divide one by two
## PowerExpression
```python
PowerExpression(self, left=None, right=None)
```
Raise one to the power of two
## ConstantExpression
```python
ConstantExpression(self, value=None)
```
A Constant value node, where the value is accessible as `node.value`
## AbsExpression
```python
AbsExpression(self, child=None, operatorOnLeft=True)
```
Evaluates the absolute value of an expression.
## SgnExpression
```python
SgnExpression(self, child=None, operatorOnLeft=True)
```

### operate
```python
SgnExpression.operate(self, value)
```

Determine the sign of an value
@returns {Number} -1 if negative, 1 if positive, 0 if 0

### differentiate
```python
SgnExpression.differentiate(self, by_variable)
```

<pre>
        f(x) = sgn( g(x) )
        d( f(x) ) = 0
</pre>
Note: in general sgn'(x) = 2δ(x) where δ(x) is the Dirac delta function

