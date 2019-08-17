from typing import Optional, Type

import gym
import numpy as np
import plac
import tensorflow as tf
from gym import error, spaces, utils
from gym.envs.registration import register
from gym.utils import seeding

from mathy.agent.controller import MathModel
from mathy.agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_MOVE_MASK,
)
from mathy.agent.training.mcts import MCTS
from mathy.core.expressions import MathTypeKeysMax
from mathy.envs.complex_term_simplification import MathyComplexTermSimplificationEnv
from mathy.envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from mathy.mathy_env import MathyEnv, MathyEnvTimeStep
from mathy.mathy_env_state import MathyEnvState
from mathy.rules.rule import ExpressionChangeRule
from mathy.util import is_terminal_transition


class MaskedDiscrete(spaces.Discrete):
    r"""A masked discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    Example::
        >>> MaskedDiscrete(3, mask=(1,1,0))
    """
    mask: np.array

    def __init__(self, n, mask):
        assert isinstance(mask, (tuple, list))
        assert len(mask) == n
        self.mask = np.array(mask)
        super(MaskedDiscrete, self).__init__(n)

    def sample(self):
        probability = self.mask / np.sum(self.mask)
        return self.np_random.choice(self.n, p=probability)


class MathyGymEnv(gym.Env):
    """"""

    metadata = {"render.modes": ["terminal"]}
    mathy: MathyEnv
    state: Optional[MathyEnvState]
    problem: Optional[str]
    env_class: Type[MathyEnv]
    env_problem_args: Optional[dict]
    last_action: int
    last_change: Optional[ExpressionChangeRule]

    def __init__(
        self,
        env_class: Type[MathyEnv] = MathyEnv,
        env_problem_args: Optional[dict] = None,
        **env_kwargs,
    ):
        self.mathy = env_class(*env_kwargs)
        self.env_class = env_class
        self.env_problem_args = env_problem_args
        self.last_action = -1
        self.last_change = None
        max_problem_types = 64
        max_nodes = 1024
        max_actions = self.mathy.action_size
        vector_width = 9  # two neighbor window extractions (1 -> 3 -> 9)
        self.state = None
        self.problem = None
        self.vectors_shape = (max_nodes, vector_width)
        self.action_space = MaskedDiscrete(max_actions, [1] * max_actions)
        self.observation_space = spaces.Dict(
            {
                FEATURE_LAST_RULE: spaces.Box(
                    low=0, high=max_actions, shape=(1,), dtype=np.int16
                ),
                FEATURE_NODE_COUNT: spaces.Box(
                    low=0, high=max_nodes, shape=(1,), dtype=np.int16
                ),
                FEATURE_PROBLEM_TYPE: spaces.Box(
                    low=0, high=max_problem_types, shape=(1,), dtype=np.int16
                ),
                FEATURE_FWD_VECTORS: spaces.Box(
                    low=0,
                    high=MathTypeKeysMax,
                    shape=self.vectors_shape,
                    dtype=np.int16,
                ),
                FEATURE_BWD_VECTORS: spaces.Box(
                    low=0,
                    high=MathTypeKeysMax,
                    shape=self.vectors_shape,
                    dtype=np.int16,
                ),
                FEATURE_LAST_FWD_VECTORS: spaces.Box(
                    low=0,
                    high=MathTypeKeysMax,
                    shape=self.vectors_shape,
                    dtype=np.int16,
                ),
                FEATURE_LAST_BWD_VECTORS: spaces.Box(
                    low=0,
                    high=MathTypeKeysMax,
                    shape=self.vectors_shape,
                    dtype=np.int16,
                ),
                FEATURE_MOVE_MASK: spaces.Box(
                    low=0, high=1, shape=(2, 2), dtype=np.int16
                ),
            }
        )

    @property
    def action_size(self) -> int:
        if self.state is not None:
            return self.mathy.get_agent_actions_count(self.state)
        return self.mathy.action_size

    def step(self, action):
        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        observation = self._observe(self.state)
        info = {"transition": transition}
        done = is_terminal_transition(transition)
        self.last_action = action
        self.last_change = change
        return observation, transition.reward, done, info

    def reset(self):
        self.last_action = -1
        self.last_change = None
        self.state, self.problem = self.mathy.get_initial_state(self.env_problem_args)
        return self._observe(self.state)

    def _observe(self, state: MathyEnvState) -> MathyEnvTimeStep:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state."""
        action_mask = self.mathy.get_valid_moves(state)
        observation = state.to_input_features(action_mask, True)
        # Update masked action space
        self.action_space.n = self.mathy.get_agent_actions_count(state)
        self.action_space.mask = action_mask
        return observation

    def render(self, mode="terminal"):
        action_name = "initial"
        token_index = -1
        if self.last_action != -1 and self.last_change is not None:
            action_index, token_index = self.mathy.get_action_indices(self.last_action)
            action_name = self.mathy.actions[action_index].name
        else:
            print(f"Problem: {self.state.agent.problem}")
        self.mathy.print_state(
            self.state, action_name[:25].lower(), token_index, change=self.last_change
        )


class MathyGymPolyEnv(MathyGymEnv):
    def __init__(self, difficulty: int = 3):
        super(MathyGymPolyEnv, self).__init__(
            env_class=MathyPolynomialSimplificationEnv,
            env_problem_args={"difficulty": difficulty},
        )


__mcts: Optional[MCTS] = None
__model: Optional[MathModel] = None


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


def main():
    env = gym.make(
        "mathy-v0",
        env_class=MathyPolynomialSimplificationEnv,
        env_problem_args={"difficulty": 5},
    )
    episodes = 10
    print_every = 2
    solved = 0
    failed = 0
    agent = "model"
    agent = "random"
    agent = "mcts"
    for i_episode in range(episodes):

        if agent == "mcts":
            mcts_start_problem(env)
        elif agent == "model":
            mathy_load_model(env)
        print_problem = i_episode % print_every == 0
        observation = env.reset()
        if print_problem:
            env.render()
        for t in range(100):
            if agent == "mcts":
                action = mcts_choose_action(env)
            elif agent == "random":
                action = env.action_space.sample()
            elif agent == "model":
                action = nn_choose_action(env)
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
    print(
        f"Finished ({episodes}) with agent ({agent})\n"
        f"Solved ({solved}) problem and Failed ({failed})"
    )
    if agent == "mcts":
        mcts_cleanup(env)
    elif agent == "model":
        mathy_free_model()
    env.close()


if __name__ == "__main__":
    register(id="mathy-v0", entry_point="gym_env:MathyGymEnv")
    plac.call(main)
