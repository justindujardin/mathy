from datetime import datetime
from typing import List, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import plac
from tqdm import trange

from mathy.agents.a3c import A3CAgent, BaseConfig
from mathy.agents.mcts import MCTS
from mathy.env.gym import MathyGymEnv

__mcts: Optional[MCTS] = None
__agent: Optional[A3CAgent] = None


def mathy_load_a3c(env_name: str, gym_env: MathyGymEnv, model: str):
    import tensorflow as tf

    global __agent
    if __agent is None:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tf.compat.v1.logging.set_verbosity("CRITICAL")
        if model is None:
            raise ValueError("model is none, must be specified")
        args = BaseConfig(model_dir=model)
        __agent = A3CAgent(args)


def mathy_free_a3c():
    global __agent
    if __agent is not None:
        __agent = None


def a3c_choose_action(gym_env: MathyGymEnv, last_observation):
    global __agent
    assert __agent is not None, "A3C agent must be initialized with: `mathy_load_a3c`"
    assert (
        gym_env.mathy is not None and gym_env.state is not None
    ), "MathyGymEnv has invalid MathyEnv or MathyEnvState members"
    return __agent.choose_action(gym_env, gym_env.state, last_observation)


def run():

    a3c_agent = "training/xfer_grouping"

    agents = ["a3c", "random"]
    topics = ["poly", "binomial", "complex", "poly-combine", "poly-commute"]
    # topics = ["complex"]

    difficulties = ["easy", "normal"]
    # difficulties = ["hard"]

    a3c_values: List[float] = []
    random_values: List[float] = []
    labels: List[str] = []
    for topic in topics:
        for difficulty in difficulties:
            labels.append(f"{topic}-{difficulty}")
            for agent in agents:
                label = f"{agent} {topic}-{difficulty}"
                env_name = f"mathy-{topic}-{difficulty}-v0"
                label = f"{label}"
                if agent == "a3c":
                    a3c_values.append(
                        main(env_name, agent, label=label, model=a3c_agent)
                    )
                    pass
                elif agent == "random":
                    random_values.append(main(env_name, agent, label=label))
                    pass

    a3c_values.reverse()
    random_values.reverse()
    labels.reverse()

    # data to plot
    n_groups = len(topics) * len(difficulties)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.54

    plt.barh(index, a3c_values, bar_width, alpha=opacity, color="b", label="a3c")

    plt.barh(
        index + bar_width * 2,
        random_values,
        bar_width,
        alpha=opacity,
        color="g",
        label="random",
    )

    diffs = "_".join(difficulties)
    title = f"agents_eval_{diffs}.png"
    plt.ylabel("Env")
    plt.xlabel("Win Percentage")
    plt.title(title)
    plt.yticks(index + bar_width * 2, labels)
    plt.legend()

    day_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"images/{day_time}{title}.png", bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def main(
    env_name: str,
    agent: str,
    label: Optional[str] = None,
    model: Optional[str] = None,
    print_every: Optional[int] = None,
) -> float:
    """Evaluate an agent in an env and return the win percentage"""
    env = gym.make(env_name)
    episodes = 100
    max_steps = 100
    solved = 0
    failed = 0

    def win_pct() -> float:
        return float("%.3f" % (float((episodes - failed) / episodes * 100)))

    with trange(episodes) as progress_bar:
        progress_bar.set_description_str(label if label is not None else agent)
        progress_bar.set_postfix(solve=solved, fail=failed, win_pct=win_pct())
        for i_episode in progress_bar:
            if agent == "a3c":
                if model is None:
                    raise ValueError("model must be specified")
                mathy_load_a3c(env_name, env, model)
            print_problem = False
            if print_every is not None:
                print_problem = i_episode % print_every == 0
            if agent == "a3c" and __agent is not None:
                __agent.global_model.embedding.reset_rnn_state()
                observation = env.reset(rnn_size=__agent.args.lstm_units)
            else:
                observation = env.reset()

            last_observation = None
            if print_problem:
                env.render()
            for t in range(max_steps):
                if agent == "random":
                    action = env.action_space.sample()
                elif agent == "a3c":
                    action = a3c_choose_action(env, last_observation)
                else:
                    raise EnvironmentError(f"unknown agent: {agent}")
                observation, reward, done, info = env.step(action)
                if print_problem:
                    env.render()
                if not done:
                    continue
                # Episode is over
                if reward > 0.0:
                    solved += 1
                else:
                    failed += 1
                break
            progress_bar.set_postfix(solve=solved, fail=failed, win_pct=win_pct())
    if agent == "a3c":
        mathy_free_a3c()
    env.close()
    return win_pct()


if __name__ == "__main__":
    plac.call(run)
