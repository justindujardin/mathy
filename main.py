# coding: utf8
"""Executing training and evaluation against the agent curriculum, automatically progressing
to the next level as the agent gets better.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
import tempfile
import numpy
import plac
from mathzero.training.lesson_runner import lesson_runner
from curriculum.combine_like_terms import lessons
import tensorflow as tf


tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    agent_name=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    )
)
def main(agent_name=None):
    if agent_name is None:
        agent_name = next(tempfile._get_candidate_names())
        print(
            "\n\nWARNING: no agent_name specified. The agent will use a random name: {}.\n\n".format(
                agent_name
            )
        )
    counter = 0
    while True:
        print("[Lesson:{}]".format(counter))
        counter = counter + 1
        lesson_runner(
            agent_name, lessons, parallel=True, dev_mode=False, skip_completed=False
        )


if __name__ == "__main__":
    plac.call(main)
