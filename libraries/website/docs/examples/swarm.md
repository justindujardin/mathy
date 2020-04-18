Mathy can solve problems using a planning only algorithm that does not require a pretrained model to be installed.

It comes with support for planning using a swarming algorithm from the [fragile](https://github.com/FragileTech/fragile){target=\_blank} library.

The basic idea behind fragile's swarm planning is to simulate a bunch of different possible actions at the same time, then select the most rewarding one to take.

In practice, the swarm planning algorithm can solve almost all of the mathy environments with little effort.

## Solve Many Tasks

Because the swarm planning algorithm doesn't require training, we can apply it to any task that Mathy exposes, and expect to see a decent result.

```Python

{!./snippets/examples/swarm_random_task.py!}

```

## Generate Training Datasets

Fragile has built-in support for generating batched datasets that can be used to train ML models. A basic example goes like this:

```Python

{!./snippets/examples/swarm_data_generation.py!}

```
