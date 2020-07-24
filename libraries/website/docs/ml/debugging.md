Debugging machine learning models is difficult at the best of times.

Mathy integrates with great tools to help make debugging easier.

## Visual Studio Code

Mathy works with Visual Studio code to allow interactive debugging of your models.

When combined with Tensorflow 2.0's eager-mode execution, you can not only debug, but also develop your models from inside the debugger.

### Installation

First download [Visual Studio Code](https://code.visualstudio.com/){target=\_blank} if you don't have it.

It's a free code editor with great features.

### Run Configurations

In order to launch Mathy in the debugger, we add a new "Run Configuration"

The JSON blob we want to add looks like this:

```json
{
  "name": "Train Mathy Agent",
  "type": "python",
  "request": "launch",
  "justMyCode": false,
  "module": "mathy",
  "args": ["train", "poly", "training/dev_agent", "--workers=1"]
}
```

By opening up `launch.json` we can insert the new configuration.

!!! tip "How to edit `launch.json`"

    Click the "Debug and Run" tab on the left side of the application. It looks like a bug icon.

    Near the top of the app it will say "Debug and Run" with a green arrow icon, and a cog icon next to it.

    Click on the cog icon to open `launch.json`

After adding the run configuration, we can set a breakpoint and launch the training process.

<img mathy-logo src="/img/set_a_breakpoint_and_debug.png" alt="Set a breakpoint and launch the debugger">

In the image above there are three points to pay attention to:

1. **To left of any code line** you can click and a **dot** will appear indicating that a breakpoint is set.
2. The **Debug and Run** side bar element in the VSCode app shows the **Run Bar**
3. The **Run Bar** element is a drop-down menu that now contains your launch item

Clicking on the **green arrow** next to the new launch configuration will start the training process.

By adding a breakpoint, when Mathy gets to the line of code we specify, it will stop executing and bring the code editor into view.

While the program is stopped in the debugger we can experiment and debug code to figure out why something is going wrong.

## Tensorboard

Mathy agents export a rich set of information to Tensorboard whenever you do agent training.

Tensorboard is a tool developed by Google to analyze Tensorflow models.

### Installation

```bash
pip install tensorboard
```

### Usage

Start a training loop:

```bash
mathy train a3c poly ./training/my_model
```

While the loop is running, execute tensorboard in a separate terminal:

```bash
tensorboard --logdir=./training/my_model
```

It will output a link to the webpage. This is usually [http://localhost:6006](http://localhost:6006){target=\_blank}

### Graphs

The mathy model graph is available in the "Graphs" tab of the Tensorboard UI:

<img mathy-logo src="/img/tb_graphs.png" alt="View model graphs in Tensorboard">

Here you can view the connections between layers in the model, and better understand the architecture your model implements.

### Scalars

The scalars tab tracks vital metrics like the average episode reward, and all the losses that are used to compute gradients.

<img mathy-logo src="/img/tb_scalars.png" alt="View model scalars in Tensorboard">

### Histograms

The histograms tab shows the change over time trainable weights. It can be useful for identifying if your model is not learning because of a bad architecture.

<img mathy-logo src="/img/tb_histograms.png" alt="View model histograms in Tensorboard">

### Problem Texts

The "text" tab receives reports of which problems are solved properly and which are not.

It outputs the input/output texts as well as the problem complexity.

<img mathy-logo src="/img/tb_text.png" alt="View model problem texts in Tensorboard">

## Snakeviz

### Installation

Which you can install if it's not already present:

```bash
pip install snakeviz
```

### Usage

If you use the `--profile` flag when training from the CLI, Mathy will save performance profiles for each worker instance on exit.

```bash
mathy train a3c poly ./training/my_model --profile
```

The profiler output files can be loaded in **[Snakeviz](https://jiffyclub.github.io/snakeviz/){target=\_blank}**

Then view the output from the previous run:

```bash
snakeviz ./training/my_model/worker_0.profile
```

Running the above command will launch a webpage on your local system.

<img mathy-logo src="/img/snakeviz_profile.png" alt="View agent performance profile in Skakeviz">
