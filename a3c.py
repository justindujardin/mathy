from mathy.a3c import A3CAgent, A3CArgs
from mathy import gym  # noqa
from typing import Optional
import plac

import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    env_name=("Initial environment name", "positional", None, str),
    transfer_from=(
        "Transfer weights from another model by its folder path",
        "positional",
        None,
        str,
    ),
    train=("Set when training is desired", "flag", False, bool),
)
def main(
    env_name="mathy-poly-03-v0",
    transfer_from: Optional[str] = None,
    train: bool = False,
):
    args = A3CArgs(train=train, update_freq=10)
    agent = A3CAgent(args, env_name=env_name, init_model=transfer_from)
    if train:
        agent.train()
    else:
        agent.play(loop=True)

    pass


if __name__ == "__main__":
    plac.call(main)
