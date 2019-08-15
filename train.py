# coding: utf8
"""Train a mathy model on a given input set, then run a lesson evaluation and exit"""
import os
from pathlib import Path
from shutil import copyfile

import plac
import tensorflow as tf

from mathy.agent.controller import MathModel
from mathy.agent.training.math_experience import (
    MathExperience,
    balanced_reward_experience_samples,
)
from mathy.mathy_env_state import INPUT_EXAMPLES_FILE_NAME
from mathy.mathy_env import MathyEnv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    examples_file=(
        "The location of a JSONL file with observations to train the model on",
        "positional",
        None,
        str,
    ),
    transfer_from=(
        "Transfer weights from another model to initialize the training model",
        "positional",
        None,
        str,
    ),
    learning_rate=("The learning rate to use when training", "option", "lr", float),
    epochs=("The number of training epochs", "option", "e", int),
    dropout=("The dropout to apply to output predictions", "option", "d", float),
)
def main(
    model_dir,
    examples_file,
    transfer_from=None,
    epochs=10,
    learning_rate=3e-4,
    dropout=0.2,
):
    train_all = True
    train_number = 2048 if not train_all else 1e6
    controller = MathyEnv(verbose=True)
    input_examples = Path(examples_file)
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        print("Making model_dir: {}".format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
    if input_examples.is_file():
        print("Copying examples into place: {}".format(model_dir))
        train_dir = model_dir / "train"
        if not train_dir.is_dir():
            train_dir.mkdir(parents=True, exist_ok=True)
        copyfile(str(input_examples), model_dir / "train" / INPUT_EXAMPLES_FILE_NAME)

    mathy = MathModel(
        controller.action_size,
        model_dir,
        init_model_dir=transfer_from,
        init_model_overwrite=True,
        learning_rate=learning_rate,
        dropout=dropout,
        long_term_size=train_number,
    )
    experience = MathExperience(mathy.model_dir)
    mathy.start()
    mathy.epochs = epochs
    train_examples = mathy.train(
        experience.short_term,
        experience.long_term,
        train_all=train_all,
        sampling_fn=balanced_reward_experience_samples,
    )
    experience.write_training_set(train_examples)
    print("Complete. Bye!")
    mathy.stop()
    exit(0)


if __name__ == "__main__":
    plac.call(main)
