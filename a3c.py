import numpy as np
import tensorflow as tf
import random

random.seed(1337)
np.random.seed(1337)
tf.random.set_seed(1337)

from mathy.agents.base_config import A3CConfig
from mathy.agents.a3c import A3CAgent
from mathy.envs import gym  # noqa
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
    strategy=("The action selection strategy to use", "option", None, str),
    embedding_units=(
        "Number of dimensions to use for token embeddings",
        "option",
        None,
        int,
    ),
    lstm_units=("Number of dimensions to use for LSTM layers", "option", None, int),
    max_eps=("Maximum number of episodes to run", "option", None, int),
    show=("Show the agents step-by-step directions", "flag", False, bool),
    profile=("Set to gather profiler outputs for the A3C workers", "flag", False, bool),
    evaluate=("Set when evaluation is desired", "flag", False, bool),
)
def main(
    topics: str,
    model_dir: str,
    transfer_from: Optional[str] = None,
    workers: int = cpu_count(),
    units: int = 32,
    embedding_units: int = 512,
    lstm_units: int = 128,
    strategy: str = "a3c",
    difficulty: Optional[str] = None,
    profile: bool = False,
    max_eps: Optional[int] = None,
    show: bool = False,
    evaluate: bool = False,
):
    topics_list = topics.split(",")
    args = A3CConfig(
        verbose=True,
        train=not evaluate,
        difficulty=difficulty,
        action_strategy=strategy,
        topics=topics_list,
        lstm_units=lstm_units,
        units=units,
        embedding_units=embedding_units,
        model_dir=model_dir,
        init_model_from=transfer_from,
        num_workers=workers,
        profile=profile,
        print_training=show,
    )
    if max_eps is not None:
        args.max_eps = max_eps

    agent = A3CAgent(args)
    if not evaluate:
        agent.train()
    else:
        agent.play(loop=True)


if __name__ == "__main__":
    plac.call(main)
