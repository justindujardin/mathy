# mathy.core.rule

## BaseRule
```python
BaseRule(self, /, *args, **kwargs)
```
Basic rule class that visits a tree with a specified visit order.
### find_node
```python
BaseRule.find_node(self, expression:mathy.core.expressions.MathExpression)
```
Find the first node that can have this rule applied to it.
### find_nodes
```python
BaseRule.find_nodes(self, expression:mathy.core.expressions.MathExpression)
```

Find all nodes in an expression that can have this rule applied to them.
Each node is marked with it's token index in the expression, according to
the visit strategy, and stored as `node.r_index` starting with index 0

## ExpressionChangeRule
```python
ExpressionChangeRule(self, rule, node=None)
```
Object describing the change to an expression tree from a rule transformation
### save_parent
```python
ExpressionChangeRule.save_parent(self, parent=None, side=None)
```
Note the parent of the node being modified, and set it as the parent of the
rule output automatically.
### done
```python
ExpressionChangeRule.done(self, node)
```
Set the result of a change to the given node. Restore the parent
if `save_parent` was called
