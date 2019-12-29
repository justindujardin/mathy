Mathy provides an on-policy learning agent (a3c) that can be trained on a modern desktop CPU using python's [threading APIs](https://docs.python.org/3.6/library/threading.html#module-threading){target=\_blank}. For simple tasks the A3C agent trains quickly, but for complex tasks it can require long training periods, and may not find reasonable solutions to difficult problems. For more difficult problems, the [zero agent](/ml/zero) performs a tree search that requires more computation, but finds reasonable solutions in large search spaces.

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

The coordinator/worker architecture used by A3C has a few features that stabilize training and allow it to quickly find solutions to some challenging tasks. By using many workers, the **diversity** of gathered training data goes up which forces the model to make predictions from a more diverse set of inputs.

??? question "Why reset the local model after updating the global model from it?"

        When a local model applies updates to the global model it does not set the weights of the model directly, rather it
        gathers gradient updates and applies them using some optimization process. This varies based on the optimizer you use
        and so it's useful to think about it as a "merge" process, where the global model is updated _partially_ by the updates
        from the local model, so it's necessary to grab all the updates from the global model afterward.

!!! info "Help Wanted - Parallelizing A3C updates"

        One thing that is challenging about A3C is that the workers all need to push their gradients
        to a shared "global" model. Tensorflow doesn't make this easy to do across process boundaries
        so the A3C implementation is strictly multi-threaded and has limited ability to scale.

        As a workaround for the inability to use multiprocessing, a hyperparameter "worker_wait" is
        defined by the agent configuration, and each worker that isn't the main (worker 0) will wait
        that many milliseconds between each action it attempts to take. This allows you to run more
        workers than you have cores. The overall number of examples gained may not be greater using this
        trick, but the diversity of the data gathered should be.

        If you would like to help out with making the A3C implementation scaling using multiprocessing
        [open an issue here](https://github.com/justindujardin/mathy/issues/new?title=A3CMultiprocessing){target=\_blank}

## Examples

The A3C agent can be interacted with via the CLI or the API directly.

### Training

You can import the required bits and train an A3C agent using your own custom python code:

```python
{!./snippets/ml/a3c_training.py!}
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

The output files can be visualized using **[Snakeviz](https://jiffyclub.github.io/snakeviz/){target=\_blank}**:

You can install Snakeviz if it's not already present:

```bash
pip install snakeviz
```

You can view the output from the previous run:

```bash
snakeviz /the_folder_my_model_is_in/worker_0.profile
```

Running the above command will launch a webpage on your local system.

The webpage has performance information about the training run that just finished.

Clicking on the various functions will expand them further.

You can use this "drilling in" to find spots of code that may be using lots of time when they should not be.

<img mathy-logo src="/img/snakeviz_profile.png" alt="View agent performance profile in Skakeviz">
