import numpy as np
import tensorflow as tf

np.random.seed(1337)
tf.random.set_seed(1337)

from mathy.a3c import A3CAgent, A3CArgs
from mathy import gym  # noqa
from typing import Optional
import plac
from multiprocessing import cpu_count

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    topics=(
        "Comma separated topic names to work problems from",
        "positional",
        None,
        str,
    ),
    model_dir=(
        "The folder to save the model at, e.g. 'training/polynomials'",
        "positional",
        None,
        str,
    ),
    transfer_from=(
        "Transfer weights from another model by its model path",
        "positional",
        None,
        str,
    ),
    difficulty=(
        "Set to force a particular difficulty rather than adjusting by performance",
        "option",
        None,
        str,
    ),
    workers=(
        "Number of worker threads to use. More increases diversity of exp",
        "option",
        None,
        int,
    ),
    units=(
        "Number of dimensions to use for math vectors and model dimensions",
        "option",
        None,
        int,
    ),
    lstm_units=("Number of dimensions to use for LSTM layers", "option", None, int),
    profile=("Set to gather profiler outputs for the A3C workers", "flag", False, bool),
    evaluate=("Set when evaluation is desired", "flag", False, bool),
)
def main(
    topics: str,
    model_dir: str,
    transfer_from: Optional[str] = None,
    workers: int = cpu_count(),
    units: int = 64,
    embedding_units: int = 512,
    lstm_units: int = 32,
    difficulty: Optional[str] = None,
    profile: bool = False,
    evaluate: bool = False,
):
    topics_list = topics.split(",")
    args = A3CArgs(
        verbose=True,
        update_freq=8,
        train=not evaluate,
        difficulty=difficulty,
        topics=topics_list,
        lstm_units=lstm_units,
        units=units,
        embedding_units=embedding_units,
        model_dir=model_dir,
        init_model_from=transfer_from,
        num_workers=workers,
        profile=profile,
    )
    agent = A3CAgent(args)
    if not evaluate:
        agent.train()
    else:
        agent.play(loop=True)

    pass


if __name__ == "__main__":
    plac.call(main)
