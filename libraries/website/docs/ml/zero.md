## Zero

The MCTS and Neural Network powered (Zero) agent is inspired by the work of Google's DeepMind and their AlphZero board-game playing AI. It uses a Monte Carlo Tree Search algorithm to produce quality actions that are unbiased by things like Actor/Critic errors.

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

## Debugging with Visual Studio Code

For [Visual Studio Code](https://code.visualstudio.com/){target=\_blank}, you can edit your `launch.json` file to add a configuration for launching Mathy with the debugger attached:

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

    On windows/linux you press the keyboard combination `CTRL+SHIFT+P`

    On mac computers you press `CMD+SHIFT+P` at the same time

    When you press the combination properly, a small autocomplete input will appear near the top of the editor.

    Begin typing "launch.json" and you'll see an entry that says something like **Debug: Open launch.json**

    By selecting that menu option a new editor window will be opened with the contents of `launch.json`

After adding the run configuration to `launch.json` file, you can set a breakpoint and launch the training process, stopping the program at specific points to look at its brain.

<img mathy-logo src="/img/set_a_breakpoint_and_debug.png" alt="Set a breakpoint and launch the debugger">

In the image above there are three points to pay attention to:

1. **To left of any code line** you can click and a **dot** will appear indicating that a breakpoint is set.
2. The **Debug and Run** side bar element in the VSCode app shows the **Run Bar**
3. The **Run Bar** element is a drop-down menu that now contains your launch item

Clicking on the **green arrow** next to the new launch configuration will start the training process.

As the training starts, the **debugger** will **attach** itself to the code and pause the app when our breakpoint tagged lines of code are run.
