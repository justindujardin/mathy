from datetime import datetime
from typing import List, Optional

import gym
import numpy as np
import plac
import tensorflow as tf
from gym.envs.registration import register
from tqdm import trange

from mathy.a3c import A3CAgent, A3CArgs
from mathy.agents.tensorflow.controller import MathModel
from mathy.agents.tensorflow.training.mcts import MCTS
from mathy.gym import MathyGymEnv

__mcts: Optional[MCTS] = None
__model: Optional[MathModel] = None
__agent: Optional[A3CAgent] = None


def mathy_load_model(gym_env: MathyGymEnv):
    global __model
    if __model is None:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tf.compat.v1.logging.set_verbosity("CRITICAL")
        __model = MathModel(gym_env.mathy.action_size, "agents/ablated")
        __model.start()


def mathy_free_model():
    global __model
    if __model is not None:
        __model.stop()
        __model = None


def mathy_load_a3c(env_name: str, gym_env: MathyGymEnv, model: str):
    global __agent
    if __agent is None:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tf.compat.v1.logging.set_verbosity("CRITICAL")
        args = A3CArgs(model_dir=model)
        __agent = A3CAgent(args)


def mathy_free_a3c():
    global __agent
    if __agent is not None:
        __agent = None


def a3c_choose_action(gym_env: MathyGymEnv):
    global __agent
    assert __agent is not None, "A3C agent must be initialized with: `mathy_load_a3c`"
    assert (
        gym_env.mathy is not None and gym_env.state is not None
    ), "MathyGymEnv has invalid MathyEnv or MathyEnvState members"
    return __agent.choose_action(gym_env, gym_env.state)


def mcts_cleanup(gym_env: MathyGymEnv):
    global __mcts
    assert (
        __mcts is not None and __model is not None
    ), "MCTS search is already destroyed"
    mathy_free_model()
    __mcts = None


def mcts_start_problem(gym_env: MathyGymEnv):
    global __mcts, __model
    num_rollouts = 500
    epsilon = 0.0
    mathy_load_model(gym_env)
    assert __model is not None
    __mcts = MCTS(
        env=gym_env.mathy,
        model=__model,
        cpuct=0.0,
        num_mcts_sims=num_rollouts,
        epsilon=epsilon,
    )


def mcts_choose_action(gym_env: MathyGymEnv) -> int:
    global __mcts, __model
    assert (
        __mcts is not None and __model is not None
    ), "MCTS search must be initialized with: `mcts_start_problem`"
    assert (
        gym_env.mathy is not None and gym_env.state is not None
    ), "MathyGymEnv has invalid MathyEnv or MathyEnvState members"
    pi = __mcts.getActionProb(gym_env.state, temp=0.0)
    action = gym_env.action_space.np_random.choice(len(pi), p=pi)
    return action


def nn_choose_action(gym_env: MathyGymEnv) -> int:
    global __model
    assert __model is not None, "MathModel must be initialized with: `mathy_load_model`"
    assert (
        gym_env.mathy is not None and gym_env.state is not None
    ), "MathyGymEnv has invalid MathyEnv or MathyEnvState members"

    # leaf node
    pi_mask = gym_env.action_space.mask
    num_valids = len(pi_mask)
    pi, _ = __model.predict(gym_env.state, pi_mask)
    pi = pi.flatten()
    # Clip any predictions over batch-size padding tokens
    if len(pi) > num_valids:
        pi = pi[:num_valids]
    # mask out invalid moves from the prediction by multiplying by
    # the pi_mask which is filled with 0 or 1 based on if the action
    # at that index is valid for the current environment state.
    min_p = abs(np.min(pi))
    pi = np.array([p + min_p for p in pi])
    pi *= pi_mask
    # In case we masked out values, we renormalize to sum to 1.
    pi_sum = np.sum(pi)
    if pi_sum > 0:
        pi /= pi_sum
    action = np.random.choice(len(pi), p=pi)
    # action = np.argmax(pi)
    return action


def run():
    import numpy as np
    import matplotlib.pyplot as plt

    a3c_agent = "training/mtl_one_bucket"

    agents = ["a3c", "random"]
    topics = ["poly", "binomial", "complex", "poly-blockers", "poly-grouping"]
    # topics = ["complex"]

    difficulties = ["easy", "normal", "hard"]
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
            if agent == "mcts":
                mcts_start_problem(env)
            elif agent == "model":
                mathy_load_model(env)
            elif agent == "a3c":
                if model is None:
                    raise ValueError("model must be specified")
                mathy_load_a3c(env_name, env, model)
            print_problem = False
            if print_every is not None:
                print_problem = i_episode % print_every == 0
            observation = env.reset()
            if print_problem:
                env.render()
            for t in range(max_steps):
                if agent == "mcts":
                    action = mcts_choose_action(env)
                elif agent == "random":
                    action = env.action_space.sample()
                elif agent == "model":
                    action = nn_choose_action(env)
                elif agent == "a3c":
                    action = a3c_choose_action(env)
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
    if agent == "mcts":
        mcts_cleanup(env)
    elif agent == "model":
        mathy_free_model()
    elif agent in ["a3c", "a3c-greedy"]:
        mathy_free_a3c()
    env.close()
    return win_pct()


if __name__ == "__main__":
    plac.call(run)
