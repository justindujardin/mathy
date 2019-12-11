# mathy.core.rule

## BaseRule
```python
BaseRule(self, /, *args, **kwargs)
```
Basic rule class that visits a tree with a specified visit order.
### apply_to
```python
BaseRule.apply_to(self, node:mathy.core.expressions.MathExpression) -> 'ExpressionChangeRule'
```
Apply the rule transformation to the given node, and return a
ExpressionChangeRule object that captures the input/output states
for the change.
### can_apply_to
```python
BaseRule.can_apply_to(self, node)
```
User-specified function that returns True/False if a rule can be
applied to a given node.

!!!warning "Performance Point"

    `can_apply_to` is called very frequently during normal operation
    and should be implemented as efficiently as possible.

### code
Short code for debug rendering. Should be two letters.
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

### name
Readable rule name used for debug rendering and description outputs
## ExpressionChangeRule
```python
ExpressionChangeRule(self, rule, node=None)
```
Object describing the change to an expression tree from a rule transformation
### done
```python
ExpressionChangeRule.done(self, node)
```
Set the result of a change to the given node. Restore the parent
if `save_parent` was called.
### save_parent
```python
ExpressionChangeRule.save_parent(self, parent=None, side=None)
```
Note the parent of the node being modified, and set it as the parent of the
rule output automatically.
