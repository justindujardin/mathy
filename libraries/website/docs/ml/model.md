## Overview

Mathy uses a model that predicts which action to take in an environment, and the scalar value of the current state.

## Model

Mathy's policy/value model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a weighted distribution over all the possible actions and value estimates for each observation.

`model:mathy.agents.model:AgentModel`

## Examples

### Call the Model

The simplest thing to do is to load a blank model and pass some data through it. This gives us a sense of how things works:

```python
{!./snippets/ml/policy_value_basic.py!}
```

### Save Model with Optimizer

Mathy's optimizer is stateful and so it has to be saved alongside the model if we want to pause and continue training later. To help with this Mathy has a function `get_or_create_policy_model`.

The helper function handles:

- Creating a folder if needed to store the model and related files
- Saving the agent hyperparameters used for training the model `model.config.json`
- Initializing and sanity checking the model by compiling and calling it with a random observation

```python
{!./snippets/ml/policy_value_serialization.py!}
```
