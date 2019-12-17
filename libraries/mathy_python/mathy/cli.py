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
    "model",
    "--model",
    default="training/alpha_piv_norm",
    help="The path to a mathy model",
)
@click.option("agent", "--agent", default="a3c", help="one of 'a3c' or 'zero'")
@click.option(
    "thinking_steps",
    "--think",
    default=3,
    help="The number of steps to think about the problem before starting an episode",
)
@click.option(
    "max_steps",
    "--steps",
    default=32,
    help="The max number of steps before the episode is over",
)
@click.argument("problem", type=str)
def cli_simplify(
    agent: str, problem: str, model: str, thinking_steps: int, max_steps: int
):
    """Simplify an input polynomial expression."""
    setup_tf_env()
    import gym
    from mathy.envs.gym import MathyGymEnv
    from colr import color

    from .agents.a3c import A3CConfig
    from .agents.action_selectors import A3CGreedyActionSelector
    from .state import observations_to_window, MathyObservation
    from .agents.policy_value_model import get_or_create_policy_model, PolicyValueModel
    from .agents.episode_memory import EpisodeMemory
    from .util import calculate_grouping_control_signal

    args = A3CConfig(
        model_dir=model,
        units=512,
        num_thinking_steps_begin=thinking_steps,
        embedding_units=512,
        lstm_units=128,
        verbose=True,
    )
    # print(args.json(indent=2))
    environment = "poly"
    difficulty = "easy"
    episode_memory = EpisodeMemory()
    env: MathyGymEnv = gym.make(f"mathy-{environment}-{difficulty}-v0")
    __model: PolicyValueModel = get_or_create_policy_model(
        args=args,
        env_actions=env.action_space.n,
        initial_state=env.initial_window(args.lstm_units),
        required=True,
    )
    last_observation: MathyObservation = env.reset_with_input(
        problem_text=problem, rnn_size=args.lstm_units, max_moves=max_steps
    )
    last_text = env.state.agent.problem
    last_action = -1
    last_reward = -1

    selector = A3CGreedyActionSelector(model=__model, episode=0, worker_id=0)

    # Set RNN to 0 state for start of episode
    selector.model.embedding.reset_rnn_state()

    # Start with the "init" sequence [n] times
    for i in range(args.num_thinking_steps_begin + 1):
        rnn_state_h = selector.model.embedding.state_h.numpy()
        rnn_state_c = selector.model.embedding.state_c.numpy()
        seq_start = env.state.to_start_observation([rnn_state_h, rnn_state_c])
        selector.model.call(observations_to_window([seq_start]).to_inputs())

    done = False
    while not done:
        env.render(args.print_mode, None)
        # store rnn state for replay training
        rnn_state_h = selector.model.embedding.state_h.numpy()
        rnn_state_c = selector.model.embedding.state_c.numpy()
        last_rnn_state = [rnn_state_h, rnn_state_c]

        # named tuples are read-only, so add rnn state to a new copy
        last_observation = MathyObservation(
            nodes=last_observation.nodes,
            mask=last_observation.mask,
            values=last_observation.values,
            type=last_observation.type,
            time=last_observation.time,
            rnn_state=last_rnn_state,
            rnn_history=episode_memory.rnn_weighted_history(args.lstm_units),
        )
        window = episode_memory.to_window_observation(last_observation)
        try:
            action, value = selector.select(
                last_state=env.state,
                last_window=window,
                last_action=last_action,
                last_reward=last_reward,
                last_rnn_state=last_rnn_state,
            )
        except KeyboardInterrupt:
            print("Done!")
            return
        except BaseException as e:
            print("Prediction failed with error:", e)
            print("Inputs to model are:", window)
            continue
        # Take an env step
        observation, reward, done, _ = env.step(action)
        rnn_state_h = selector.model.embedding.state_h.numpy()
        rnn_state_c = selector.model.embedding.state_c.numpy()
        observation = MathyObservation(
            nodes=observation.nodes,
            mask=observation.mask,
            values=observation.values,
            type=observation.type,
            time=observation.time,
            rnn_state=[rnn_state_h, rnn_state_c],
            rnn_history=episode_memory.rnn_weighted_history(args.lstm_units),
        )

        new_text = env.state.agent.problem
        grouping_change = calculate_grouping_control_signal(
            last_text, new_text, clip_at_zero=True
        )
        episode_memory.store(
            observation=last_observation,
            action=action,
            reward=reward,
            grouping_change=grouping_change,
            value=value,
        )
        if done:
            # Last timestep reward
            win = reward > 0.0
            env.render(args.print_mode, None)
            print(
                color(text="SOLVE" if win else "FAIL", fore="green" if win else "red",)
            )
            break

        last_observation = observation
        last_action = action
        last_reward = reward


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
        try:
            env.mathy.parser.parse(problem.text)
            msg.good(problem.text)
        except BaseException as error:
            msg.fail(f"Parse failed '{problem.text}':  {error}")


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
