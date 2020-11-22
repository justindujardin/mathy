Mathy can solve problems using a swarm-planning algorithm from the [fragile](https://github.com/FragileTech/fragile){target=\_blank} library that does not require a pre-trained model.

The basic idea behind fragile's swarm planning is to simulate many different possible actions simultaneously, then select the most rewarding one to take.

In practice, the swarm planning algorithm can solve almost all of the mathy environments with little effort.

## Solve Many Tasks

Because the swarm planning algorithm doesn't require training, we can apply it to any task that Mathy exposes and expect to see a decent result.

```Python

{!./snippets/examples/swarm_random_task.py!}

```

## Generate Training Datasets

Fragile has built-in support for generating batched datasets for training ML models. A basic example goes like this:

```Python

{!./snippets/examples/swarm_data_generation.py!}

```
