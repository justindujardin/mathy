Debugging machine learning models is difficult at the best of times.

Good tools can help make the job of debugging your models easier.

## Tensorboard

Mathy agents export a rich set of information to Tensorboard including:

- **Model Graph Visualizations**: Mathy uses Tensorflow's trace API to record the agent's model and export it to the Graph tab in Tensorboard.
- **Generated Problems and Outcomes**: See what kind of problems the agent is succeeding at solving and which ones it is struggling with.

## Snakeviz

Using the `--profile` flag when training frm the CLI will export A3C worker profiler dump files that can be loaded in Skakeviz
