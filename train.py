# coding: utf8
"""Train a mathy model on a given input set, then run a lesson evaluation and exit"""
import json
import os
import random
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from shutil import copyfile

import numpy
import plac
import tensorflow as tf
from colr import color

from mathy.agent.controller import MathModel
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.math_experience import MathExperience
from mathy.agent.training.mcts import MCTS
from mathy.environment_state import INPUT_EXAMPLES_FILE_NAME
from mathy.math_game import MathGame

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
)
def main(model_dir, examples_file, transfer_from=None):
    epochs = 24
    train_all = False
    train_number = 4096
    controller = MathGame(verbose=True)
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
        learning_rate=3e-4,
        dropout=0.2,
        long_term_size=train_number,
    )
    experience = MathExperience(mathy.model_dir)
    mathy.start()
    mathy.epochs = epochs
    mathy.train(experience.short_term, experience.long_term, train_all=train_all)
    print("Complete. Bye!")
    mathy.stop()
    exit(0)


if __name__ == "__main__":
    plac.call(main)
