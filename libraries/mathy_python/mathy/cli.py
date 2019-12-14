"""Mathy CLI
---

Command line application for interacting with Mathy agents and environments.
"""
from multiprocessing import cpu_count

import click


@click.group()
@click.version_option()
def cli():
    """
    Mathy - https://mathy.ai

    Command line app for training and evaluating agents that transform
    expression trees using reinforcement learning.
    """


@cli.command("contribute")
def cli_contribute():
    """Learn about ways you can contribute to Mathy's development."""
    import webbrowser

    webbrowser.open("https://mathy.ai/contributing", new=2)


@cli.command("simplify")
@click.option("agent", "--agent", default="zero", help="one of 'a3c' or 'zero'")
@click.argument("problem", type=str)
def cli_simplify(agent: str, problem: str):
    """Simplify an input polynomial expression."""
    print(f"neat: {agent} -> {problem}")

    from .a3c import main

    main(topics="poly", model_dir="training/poly", max_eps=1, show=True, evaluate=True)


@cli.command("problems")
@click.argument("environment", type=str)
@click.option(
    "difficulty",
    "--difficulty",
    default="easy",
    help="One of 'easy', 'normal', or 'hard'",
)
@click.option("number", "--number", default=25, help="The number of problems to print")
def cli_print_problems(environment: str, difficulty: str, number: int):
    """Print a set of generated problems from a given environment.

    This is useful if you when developing new environment types for
    verifying that the problems you're generating take the form you
    expect. """
    import gym
    from mathy.envs.gym import MathyGymEnv

    env: MathyGymEnv = gym.make(f"mathy-{environment}-{difficulty}-v0")

    for i in range(number):
        state, problem = env.mathy.get_initial_state(print_problem=False)
        print(problem.text)


@cli.command("train")
@click.argument("agent")
@click.argument("topics")
@click.argument("folder")
@click.option(
    "transfer",
    "--transfer",
    default=None,
    help="Location of a model to initialize the agent's weights from",
)
@click.option(
    "difficulty",
    "--difficulty",
    default=None,
    help="the difficulty of problems to generate, 'easy','normal','hard'",
)
@click.option(
    "workers",
    "--workers",
    default=cpu_count(),
    type=int,
    help="Number of worker threads to use. More increases diversity of exp",
)
@click.option(
    "strategy",
    "--strategy",
    default="a3c",
    type=str,
    help="The action selection strategy to use",
)
@click.option(
    "units",
    "--units",
    default=512,
    type=int,
    help="Number of dimensions to use for math vectors and model dimensions",
)
@click.option(
    "embeddings",
    "--embeddings",
    default=512,
    type=int,
    help="Number of dimensions to use for token embeddings",
)
@click.option(
    "rnn",
    "--rnn",
    default=128,
    type=int,
    help="Number of dimensions to use for RNN state",
)
@click.option(
    "episodes",
    "--episodes",
    default=None,
    type=int,
    help="Maximum number of episodes to run",
)
@click.option(
    "mcts_sims",
    "--mcts-sims",
    default=250,
    type=int,
    help="Number of rollouts per timestep when using MCTS",
)
@click.option(
    "show",
    "--show",
    default=False,
    is_flag=True,
    help="Show the agents step-by-step directions",
)
@click.option(
    "profile",
    "--profile",
    default=False,
    is_flag=True,
    help="Set to gather profiler outputs for workers",
)
@click.option(
    "verbose",
    "--verbose",
    default=False,
    is_flag=True,
    help="Display verbose log items",
)
def cli_train(
    agent: str,
    topics: str,
    folder: str,
    transfer: str,
    difficulty: str,
    strategy: str,
    workers: int,
    units: int,
    embeddings: int,
    rnn: int,
    profile: bool,
    episodes: int,
    mcts_sims: int,
    show: bool,
    verbose: bool,
):
    """Train an agent to solve math problems and save the model.

    Arguments:

    "agent" is one of the known agent types, either "a3c" or "zero".

    "topics" is a comma separated list of topic names to work problems from.

    "folder" is the output location to store the model and tensorboard logs.
    e.g. "/tmp/training/custom_agent/"

    """
    topics_list = topics.split(",")

    if agent == "a3c":
        setup_tf_env()

        from .agents.a3c import A3CAgent, A3CConfig

        args = A3CConfig(
            verbose=verbose,
            train=True,
            difficulty=difficulty,
            action_strategy=strategy,
            topics=topics_list,
            lstm_units=rnn,
            units=units,
            embedding_units=embeddings,
            mcts_sims=mcts_sims,
            model_dir=folder,
            init_model_from=transfer,
            num_workers=workers,
            profile=profile,
            print_training=show,
        )
        if episodes is not None:
            args.max_eps = episodes
        instance = A3CAgent(args)
        instance.train()
    elif agent == "zero":
        setup_tf_env()
        from .agents.zero import SelfPlayConfig, self_play_runner

        self_play_cfg = SelfPlayConfig(
            verbose=verbose,
            train=True,
            difficulty=difficulty,
            action_strategy=strategy,
            topics=topics_list,
            lstm_units=rnn,
            units=units,
            embedding_units=embeddings,
            mcts_sims=mcts_sims,
            model_dir=folder,
            init_model_from=transfer,
            num_workers=workers,
            profile=profile,
            print_training=show,
        )

        self_play_runner(self_play_cfg)


def setup_tf_env():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(1337)
    np.random.seed(1337)
    tf.random.set_seed(1337)

    tf.compat.v1.logging.set_verbosity("CRITICAL")


if __name__ == "__main__":
    cli()
