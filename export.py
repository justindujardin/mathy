# coding: utf8
import os
import plac
import tensorflow as tf
from pathlib import Path
from mathy.agent.controller import MathModel
from mathy import MathyEnv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    export_dir=(
        "The export path for the saved model to be placed in.",
        "positional",
        None,
        str,
    ),
)
def main(model_dir, export_dir):
    mathy = MathModel(MathyEnv().action_size, model_dir)
    export_path = Path(export_dir)
    if not export_path.is_dir():
        export_path.mkdir(parents=True, exist_ok=True)
    mathy.export(export_dir)


if __name__ == "__main__":
    plac.call(main)
