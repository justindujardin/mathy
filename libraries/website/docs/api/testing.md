# mathy.testing

## run_rule_tests
```python
run_rule_tests(name, rule_class, callback=None)
```
Load and assert about the transformations and validity of rules
based on given input examples.

When debugging a problem it can be useful to provide a "callback" function
and add a `"debug": true` value to the example in the rules json file you
want to debug. Then you set a breakpoint and step out of your callback function
into the parsing/evaluation of the debug example.

