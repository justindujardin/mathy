## Zero

The MCTS and Neural Network powered (Zero) agent is inspired by the work of Google's DeepMind and their AlphZero board-game playing AI. It uses a Monte Carlo Tree Search algorithm to produce quality actions that are unbiased by things like Actor/Critic errors.

## Examples

### Training

Use the self-play functionality to train a zero agent using the [Policy/Value](/ml/policy_value) model:

```python
{!./snippets/ml/zero_training.py!}
```

## Debugging

Mathy's Zero agent uses the Python **[multiprocessing](https://docs.python.org/3.7/library/multiprocessing.html){target=\_blank}** module to spawn a bunch of workers to execute episodes at the same time.

For training this multi-worker approach make sense, and speeds up example gathering considerably.

However, running with multiple workers breaks some modern debuggers like [Visual Studio Code](https://code.visualstudio.com/){target=\_blank}.

Because of this, the Zero agent will prefer to use the Python **[threading](https://docs.python.org/3.7/library/threading.html){target=\_blank}** module if it is configured for only **one worker**.

In this mode you can set breakpoints in the debugger to help diagnose errors.

### Visual Studio Code

For VSCode, you can edit your `launch.json` file to add a configuration for launching Mathy with the debugger attached:

```json
{
  "name": "Train Mathy Zero Agent",
  "type": "python",
  "request": "launch",
  "justMyCode": false,
  "module": "mathy",
  "args": ["train", "zero", "poly", "training/dev_zero", "--workers=1"]
}
```

!!! tip "How to edit `launch.json`"

    It can be tricky to find `launch.json` if you've never done it before, but don't worry it's simple enough.

    With VS Code open, press the keyboard keys CMD+SHIFT+P at the same time, and a small autocomplete input will appear near the top of the app.

    Begin typing "launch.json" and you'll see an entry that says something like **Debug: Open launch.json**
