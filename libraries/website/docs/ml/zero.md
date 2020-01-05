## Zero

The MCTS and Neural Network powered (Zero) agent is inspired by the work of Google's DeepMind and their [AlphZero](/about/#alphazero) board-game playing AI. It uses a [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search){target=\_blank} algorithm to produce quality actions that are unbiased by things like Actor/Critic errors.

## Multiple Process Training

Mathy's Zero agent uses the Python **[multiprocessing](https://docs.python.org/3.7/library/multiprocessing.html){target=\_blank}** module to train with many copies of the agent at the same time.

For long training runs, this multi-worker approach speeds up example gathering considerably.

Let's see what this looks like with self-play.

We'll train a zero agent using the [Policy/Value](/ml/policy_value) model:

```python
{!./snippets/ml/zero_training.py!}
```

## Single Process Training

Running multiple process training does not work great with some modern debuggers like [Visual Studio Code](https://code.visualstudio.com/){target=\_blank}.

Because of this, the Zero agent will use the Python **[threading](https://docs.python.org/3.7/library/threading.html){target=\_blank}** module if it is configured to use **only one worker**.

In this mode you can set breakpoints in the debugger to help diagnose errors.

```python
{!./snippets/ml/zero_debugging.py!}
```

## Performance Profiling

The CLI Zero agent can optionally output performance profiles.

For the CLI you do this with `--profile --num-workers=1`.

In python you pass `profile=True` and `num_workers=1` to the agent configuration.

```python
{!./snippets/ml/zero_profiling.py!}
```

Learn about how to view output profiles on the [debugging page](/ml/debugging/#snakeviz)
