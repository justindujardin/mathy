"""Mathy CLI
---

Command line application for interacting with Mathy agents and environments.
"""
from multiprocessing import cpu_count

import click
from wasabi import msg


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
@click.option(
    "swarm",
    "--swarm",
    default=False,
    is_flag=True,
    help="Use swarm solver from fragile library without a trained model",
)
@click.option(
    "model", "--model", default="mathy_alpha_sm", help="The path to a mathy model",
)
@click.option("agent", "--agent", default="a3c", help="one of 'a3c' or 'zero'")
@click.option(
    "max_steps",
    "--max-steps",
    default=20,
    help="The max number of steps before the episode is over",
)
@click.argument("problem", type=str)
def cli_simplify(agent: str, problem: str, model: str, max_steps: int, swarm: bool):
    """Simplify an input polynomial expression."""
    setup_tf_env()

    from .models import load_model
    from .api import Mathy
    from .agents.fragile import SwarmConfig

    mt: Mathy
    if swarm is True:
        mt = Mathy(config=SwarmConfig())
    else:
        try:
            mt = load_model(model)
        except ValueError:
            mt = Mathy(config=SwarmConfig())

    mt.simplify(problem=problem, max_steps=max_steps)


@cli.command("problems")
@click.argument("environment", type=str)
@click.option(
    "difficulty",
    "--difficulty",
    default="easy",
    help="One of 'easy', 'normal', or 'hard'",
)
@click.option("number", "--number", default=10, help="The number of problems to print")
def cli_print_problems(environment: str, difficulty: str, number: int):
    """Print a set of generated problems from a given environment.

    This is useful if you when developing new environment types for
    verifying that the problems you're generating take the form you
    expect. """
    import gym
    from mathy.envs.gym import MathyGymEnv

    env_name = f"mathy-{environment}-{difficulty}-v0"
    env: MathyGymEnv = gym.make(env_name)
    msg.divider(env_name)
    with msg.loading(f"Generating {number} problems..."):
        header = ("Complexity", "Is Valid", "Text")
        widths = (10, 8, 62)
        aligns = ("c", "c", "l")
        data = []
        for i in range(number):
            state, problem = env.mathy.get_initial_state(
                env.env_problem_args, print_problem=False
            )
            valid = False
            text = problem.text
            try:
                env.mathy.parser.parse(problem.text)
                valid = True
            except BaseException as error:
                text = f"parse failed for '{problem.text}' with error: {error}"
            data.append((problem.complexity, "✔" if valid else "✘", text,))
    msg.good(f"\nGenerated {number} problems!")

    print(msg.table(data, header=header, divider=True, widths=widths, aligns=aligns))


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
    default=256,
    type=int,
    help="Number of dimensions to use for math vectors and model dimensions",
)
@click.option(
    "rnn",
    "--rnn",
    default=None,
    type=int,
    help="Number of dimensions to use for token embeddings",
)
@click.option(
    "embeddings",
    "--embeddings",
    default=256,
    type=int,
    help="Number of dimensions to use for token embeddings",
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
    default=10,
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
    "self_play_problems",
    "--self-play-problems",
    default=100,
    help="The number of self-play problems per gather/training iteration",
)
@click.option(
    "training_iterations",
    "--training-iterations",
    default=10,
    help="The max number of time to perform gather/training loops for the zero agent",
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
    rnn: int,
    embeddings: int,
    profile: bool,
    episodes: int,
    mcts_sims: int,
    show: bool,
    verbose: bool,
    training_iterations: int,
    self_play_problems: int,
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
            difficulty=difficulty,
            action_strategy=strategy,
            topics=topics_list,
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
        if rnn is not None:
            args.lstm_units = rnn
        instance = A3CAgent(args)
        instance.train()
    elif agent == "zero":
        setup_tf_env(use_mp=workers > 1)
        from .agents.zero import SelfPlayConfig, self_play_runner

        self_play_cfg = SelfPlayConfig(
            verbose=verbose,
            difficulty=difficulty,
            topics=topics_list,
            units=units,
            embedding_units=embeddings,
            mcts_sims=mcts_sims,
            model_dir=folder,
            init_model_from=transfer,
            num_workers=workers,
            training_iterations=training_iterations,
            self_play_problems=self_play_problems,
            print_training=show,
            profile=profile,
        )
        if episodes is not None:
            self_play_cfg.max_eps = episodes

        self_play_runner(self_play_cfg)


def setup_tf_env(use_mp=False):
    if use_mp:
        setup_tf_env_mp()
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(1337)
    np.random.seed(1337)
    tf.random.set_seed(1337)

    tf.compat.v1.logging.set_verbosity("CRITICAL")


def setup_tf_env_mp():
    """Create a sub-process and import Tensorflow inside of it
    so that the library is loaded first in a subprocess. This is
    the hacky way we get multiprocessing to work reliably. |o_O|"""
    import multiprocessing

    def worker():
        import tensorflow as tf

        return 0

    proc = multiprocessing.Process(target=worker)
    proc.start()
    proc.join()


if __name__ == "__main__":
    cli()
