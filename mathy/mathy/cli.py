"""Mathy CLI
---

Command line application for interacting with Mathy agents and environments.
"""

import os
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
    "single_process",
    "--single-process",
    default=os.name == "nt", # fragile swarm mp is unreliable on windows
    is_flag=True,
    help="Use single-process execution with the swarm solver",
)
@click.option(
    "max_steps",
    "--max-steps",
    default=20,
    help="The max number of steps before the episode is over",
)
@click.option(
    "num_walkers",
    "--num-walkers",
    default=512,
    help="The max number of steps before the episode is over",
)
@click.argument("problem", type=str)
def cli_simplify(problem: str, max_steps: int, single_process: bool, num_walkers: int):
    """Simplify an input polynomial expression."""

    from .api import Mathy
    from .solver import SwarmConfig

    mt = Mathy(
        config=SwarmConfig(
            use_mp=not single_process, n_walkers=num_walkers, verbose=True
        )
    )
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
    import gymnasium as gym
    from mathy_envs.gym import MathyGymEnv

    env_name = f"mathy-{environment}-{difficulty}-v0"
    env: MathyGymEnv = gym.make(env_name).unwrapped  # type:ignore
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


if __name__ == "__main__":
    cli()
