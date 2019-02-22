# coding: utf8
"""Executing training and evaluation against the agent curriculum, automatically progressing
to the next level as the agent gets better.
"""
import numpy
import plac
from mathzero.training.lesson_runner import lesson_runner
from curriculum.combine_like_terms import lessons


@plac.annotations(
    agent_name=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    )
)
def main(agent_name="default"):
    lesson_runner("combine_terms_2", lessons, parallel=True, dev_mode=False)


if __name__ == "__main__":
    plac.call(main)
