Debugging machine learning models is difficult at the best of times.

Mathy integrates with great tools to help make debugging easier.

## Visual Studio Code

Mathy works with Visual Studio code to allow interactive debugging of your models.

When combined with Tensorflow 2.0's eager-mode execution, you can not only debug, but also develop your models from inside the debugger.

Docs todo...

## Tensorboard

Mathy agents export a rich set of information to Tensorboard including:

- **Model Graph Visualizations**: Mathy uses Tensorflow's trace API to record the agent's model and export it to the Graph tab in Tensorboard.
- **Generated Problems and Outcomes**: See what kind of problems the agent is succeeding at solving and which ones it is struggling with.

Whenever you do agent training, Mathy will save an extra folder in the model output call `tensorboard`

The `tensorboard` folder contains event data that can be viewed in Tensorboard:

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
