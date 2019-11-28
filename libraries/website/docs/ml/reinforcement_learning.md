
## MathExpression
```python
MathExpression(self, id=None, left=None, right=None, parent=None)
```

A Basic MathExpression node

### evaluate
```python
MathExpression.evaluate(self, context=None)
```
Evaluate the expression, resolving all variables to constant values
### differentiate
```python
MathExpression.differentiate(self, byVar)
```
Differentiate the expression by a given variable
### addClass
```python
MathExpression.addClass(self, classes)
```

Associate a class name with an expression.  This class name will be tagged on nodes
when the expression is converted to a capable output format.  See {@link `getMathML`}.

### toList
```python
MathExpression.toList(self)
```

Convert this node hierarchy into a list.
@returns {Array} Array of {@link MathExpression} visited in order

### findByType
```python
MathExpression.findByType(self, instanceType)
```

Find an expression in this tree by type.
@param {Function} instanceType The type to check for instances of
@returns {Array} Array of {@link MathExpression} that are of the given type.

### findById
```python
MathExpression.findById(self, id)
```

Find an expression by its unique ID.
@returns {MathExpression|None} The node.

### toMathML
```python
MathExpression.toMathML(self)
```
Convert this single node into MathML.
### getMathML
```python
MathExpression.getMathML(self)
```
Convert this expression into a MathML container.
### makeMLTag
```python
MathExpression.makeMLTag(self, tag, content, classes=[])
```

Make an ML tag for the given content, respecting the node's
given classes.
@param {String} tag The ML tag name to create.
@param {String} content The ML content to place inside of the tag.
@param {Array} classes An array of classes to attach to this tag.

### pathToRoot
```python
MathExpression.pathToRoot(self)
```

Generate a namespaced path key to from the current node to the root.
This key can be used to identify a node inside of a tree.

### rootClone
```python
MathExpression.rootClone(self, node=None)
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
node.  See {@link `rootClone`} for more details.

### isSubTerm
```python
MathExpression.isSubTerm(self)
```

Determine if this is a sub-term, meaning it has
a parent that is also a term, in the expression.

This indicates that a term has limited mobility,
and cannot be freely moved around the entire expression.

### getTerm
```python
MathExpression.getTerm(self)
```
Get the term that this node belongs to. Return boolean of expression
### getTerms
```python
MathExpression.getTerms(self)
```
Get any terms that are children of this node. Returns a list of expressions
## UnaryExpression
```python
UnaryExpression(self, child, operatorOnLeft=True)
```
An expression that operates on one sub-expression
## NegateExpression
```python
NegateExpression(self, child, operatorOnLeft=True)
```
Negate an expression, e.g. `4` becomes `-4`
### differentiate
```python
NegateExpression.differentiate(self, byVar)
```

<pre>
        f(x) = -g(x)
    d( f(x) ) = -( d( g(x) ) )
</pre>

## FunctionExpression
```python
FunctionExpression(self, child, operatorOnLeft=True)
```

A Specialized UnaryExpression that is used for functions.  The function name in
text (used by the parser and tokenizer) is derived from the name() method on
the class.

## BinaryExpression
```python
BinaryExpression(self, left=None, right=None)
```
An expression that operates on two sub-expressions
### getPriority
```python
BinaryExpression.getPriority(self)
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
EqualExpression.operate(self, one, two)
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
## AbsExpression
```python
AbsExpression(self, child, operatorOnLeft=True)
```
Evaluates the absolute value of an expression.
## SgnExpression
```python
SgnExpression(self, child, operatorOnLeft=True)
```

### operate
```python
SgnExpression.operate(self, value)
```

Determine the sign of an value
@returns {Number} -1 if negative, 1 if positive, 0 if 0

### differentiate
```python
SgnExpression.differentiate(self, byVar)
```

<pre>
        f(x) = sgn( g(x) )
        d( f(x) ) = 0
</pre>
Note: in general sgn'(x) = 2δ(x) where δ(x) is the Dirac delta function

