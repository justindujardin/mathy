Mathy provides an on-policy learning agent (a3c) that can be trained on a modern desktop CPU using python's [threading APIs](https://docs.python.org/3.6/library/threading.html#module-threading){target=\_blank}.

For some tasks the A3C agent trains quickly, but for more complex tasks it can require long training periods.

## Asynchronous Advantage Actor-Critic

Asynchronous Advantage Actor-Critic (A3C) is an algorithm that uses multiple workers to train a shared model. It roughly breaks down like this:

!!! info "A3C Pseudocode"

        1. create a **global** model that workers will update
        2. create n **worker** threads and pass the global model to them
        3. [worker loop]
            - create a **local** model by copying the **global** model weights
            - take up to **update_interval** actions or complete an episode
            - apply updates to the **local** model from gathered data
            - merge **local** model changes with **global** model
        4.  **done**

The coordinator/worker architecture used by A3C has a few features that stabilize training and allow it to quickly find solutions to some challenging tasks.

By using many workers, the **diversity** of training data goes up which forces the model to make predictions from a more diverse set of inputs.

## Examples

The A3C agent can be interacted with via the CLI or the API directly.

### Training

You can import the required bits and train an A3C agent using your own custom python code:

```python
{!./snippets/ml/a3c_training.py!}
```

### Training A3C+MCTS

Mathy exposes a few "action_strategy" options, including two for dealing with the combination of A3C and MCTS.

Remember that MCTS is a powerful tree search that produces better actions the more "rollouts" or "simulations" you give it.

#### MCTS Worker "0"

The action strategy "mcts_worker_0" runs **MCTS only on worker #0**, the "greedy" worker.

Because worker 0 is the one without random exploration, this means that the output and average episode rewards will all be based on a tree search, while in the background the other A3C workers will continue to use vanilla Actor/Critic action selection.

```python
{!./snippets/ml/a3c_training_mcts_worker_0.py!}
```

#### MCTS Worker "n"

The action strategy "mcts_worker_n" runs **MCTS on all workers that are not #0** (the "greedy" worker).

The idea behind this strategy was that perhaps MCTS observations would be beneficial to mix in with biased Actor/Critic observations.

Because the 0 worker still uses plain A3C, its outputs should give a good sense of how the model will perform at inference time without MCTS.

!!! warn "Warning about Performance"

    Because the A3C workers are parallelized using the python threading module, this strategy does not generally perform well with **many workers**.

    This is because python's `threading` module doesn't allow full system resource utilization when compared to the `multiprocessing` module.

    I've found that it's better to use small numbers of workers with this strategy.

```python
{!./snippets/ml/a3c_training_mcts_worker_n.py!}
```

### Training with the CLI

Once mathy is installed on your system, you can train an agent using the CLI:

```bash
mathy train a3c poly output/my_agent --show
```

!!! info "Viewing the agent training"

    You can view the agent's in-episode actions by providing the `--show` argument when using the CLI

### Performance Profiling

The CLI A3C agent accepts a `--profile` option, or a config option when the API is used.

```python
{!./snippets/ml/a3c_profiling.py!}
```

Learn about how to view output profiles on the [debugging page](/ml/debugging/#snakeviz)

## Multiprocess A3C

One thing that is challenging about A3C is that the workers all need to push their gradients
to a shared "global" model. Tensorflow doesn't make this easy to do across process boundaries
so the A3C implementation is strictly multi-threaded and has limited ability to scale.

!!! info "Help Wanted - Parallelizing A3C updates"

    If you would like to help out with making the A3C implementation scaling using multiprocessing
    [open an issue here](https://github.com/justindujardin/mathy/issues/new?title=A3CMultiprocessing){target=\_blank}

As a workaround for the inability to use multiprocessing, a hyperparameter "worker_wait" is
defined by the agent configuration, and each worker that isn't the main (worker 0) will wait
that many milliseconds between each action it attempts to take. This allows you to run more
workers than you have cores. The overall number of examples gained may not be greater using this
trick, but the diversity of the data gathered should be.
