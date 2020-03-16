from typing import List
import copy
import sys
import time
import traceback

import gym
import holoviews
import numpy as np
import srsly
from gym import spaces

from fragile.core.env import DiscreteEnv
from fragile.core.models import Bounds, DiscreteModel
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.swarm import Swarm
from fragile.core.tree import HistoryTree
from fragile.core.utils import StateDict
from fragile.core.walkers import Walkers
from fragile.dataviz import AtariViz, LandscapeViz, Summary, SwarmViz, SwarmViz1D
from mathy import MathTypeKeysMax, MathyEnvState, is_terminal_transition, EnvRewards
from mathy.envs.gym import MathyGymEnv
from plangym import ParallelEnvironment
from plangym.env import Environment
import os


# Print explored mathy states when True
verbose = False
use_mp = True
prune_tree = True
max_iters = 100
reward_scale = 5
distance_scale = 10
minimize = False
use_vis = False


class DiscreteMasked(DiscreteModel):
    def sample(
        self,
        batch_size: int,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        **kwargs,
    ) -> StatesModel:
        def random_choice_prob_index(a, axis=1):
            """Select random actions with probabilities across a batch.
            
            Source: https://stackoverflow.com/a/47722393/287335"""
            r = np.expand_dims(self.random_state.rand(a.shape[1 - axis]), axis=axis)
            return (a.cumsum(axis=axis) > r).argmax(axis=axis)

        if env_states is not None:
            # Each state is a vstack([node_ids, mask]) and we only want the mask.
            #
            # Swap columns and slice the last element to get it.
            masks = np.transpose(env_states.observs, [0, 2, 1])[:, :, -1]
            actions = random_choice_prob_index(masks)
        else:
            actions = self.random_state.randint(0, self.n_actions, size=batch_size)
        return self.update_states_with_critic(
            actions=actions, model_states=model_states, batch_size=batch_size, **kwargs
        )


class MathySwarm(Swarm):
    def calculate_end_condition(self) -> bool:
        """Stop when a walker receives a positive terminal reward."""
        max_reward = self.walkers.env_states.rewards.max()
        return max_reward > EnvRewards.WIN or self.walkers.calculate_end_condition()


class FragileMathyEnv(DiscreteEnv):
    """FragileMathyEnv is an interface between the `plangym.Environment` and a
    Mathy environment."""

    STATE_CLASS = StatesEnv

    def get_params_dict(self) -> StateDict:
        super_params = super(FragileMathyEnv, self).get_params_dict()
        params = {"game_ends": {"dtype": np.bool_}}
        params.update(super_params)
        return params

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        actions = model_states.actions.astype(np.int32)
        new_states, observs, rewards, game_ends, infos = self._env.step_batch(
            actions=actions, states=env_states.states
        )
        ends = [not inf.get("valid", False) for inf in infos]
        new_state = self.states_from_data(
            states=new_states,
            observs=observs,
            rewards=rewards,
            ends=ends,
            batch_size=len(actions),
            game_ends=game_ends,
        )
        return new_state

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        state, obs = self._env.reset()
        states = np.array([copy.deepcopy(state) for _ in range(batch_size)])
        observs = np.array([copy.deepcopy(obs) for _ in range(batch_size)])
        rewards = np.zeros(batch_size, dtype=np.float32)
        ends = np.zeros(batch_size, dtype=np.bool_)
        game_ends = np.zeros(batch_size, dtype=np.bool_)
        new_states = self.states_from_data(
            states=states,
            observs=observs,
            rewards=rewards,
            ends=ends,
            batch_size=batch_size,
            game_ends=game_ends,
        )
        return new_states


class FragileEnvironment(Environment):
    """Fragile Environment for solving Mathy problems."""

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "normal",
        wrappers=None,
        **kwargs,
    ):
        super(FragileEnvironment, self).__init__(name=name)
        self._env: MathyGymEnv = gym.make(
            f"mathy-{environment}-{difficulty}-v0",
            verbose=verbose,
            np_observation=True,
            **kwargs,
        )
        self.observation_space = spaces.Box(
            low=0, high=MathTypeKeysMax, shape=(256, 256, 1), dtype=np.uint8,
        )
        self.wrappers = wrappers
        self.init_env()

    def init_env(self):
        env = self._env
        env.reset()
        if self.wrappers is not None:
            for wrap in self.wrappers:
                env = wrap(env)
        self.action_space = spaces.Discrete(self._env.action_size)
        self.observation_space = (
            self._env.observation_space
            if self.observation_space is None
            else self.observation_space
        )

    def __getattr__(self, item):
        return getattr(self._env, item)

    def get_state(self) -> np.ndarray:
        assert self._env is not None, "env required to get_state"
        return self._env.state.to_np()

    def set_state(self, state: np.ndarray):
        assert self._env is not None, "env required to set_state"
        self._env.state = MathyEnvState.from_np(state)
        return state

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        assert self._env is not None, "env required to step"
        if state is not None:
            self.set_state(state)
        obs, reward, _, info = self._env.step(action)
        terminal = info.get("done", False)
        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, terminal, info
        return obs, reward, terminal, info

    def step_batch(
        self, actions, states=None, n_repeat_action: [int, np.ndarray] = None
    ) -> tuple:
        data = [self.step(action, state) for action, state in zip(actions, states)]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        if states is None:
            return observs, rewards, terminals, infos
        else:
            return new_states, observs, rewards, terminals, infos

    def reset(self, return_state: bool = True):
        assert self._env is not None, "env required to reset"
        obs = self._env.reset()
        if not return_state:
            return obs
        else:
            return self.get_state(), obs


def mathy_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=1)


def solve_problem(problem_env: str, problem_difficulty: str = "easy"):
    if use_mp:
        env = ParallelEnvironment(
            env_class=FragileEnvironment,
            name="mathy_v0",
            environment=problem_env,
            difficulty=problem_difficulty,
            repeat_problem=True,
        )
    else:
        env = FragileEnvironment(
            name="mathy_v0",
            environment=problem_env,
            difficulty=problem_difficulty,
            repeat_problem=True,
        )

    print(f"\n\nProblem: {env._env.state.agent.problem}")
    print(f"Type: {problem_env}\tDifficulty: {problem_difficulty}\n\n")

    # the hard difficulty problems tend to have >= 100 nodes and need more workers to find
    # solutions in such a large space since the workers can't go back in time.
    # print_every = 5 if problem_difficulty == "hard" else 10
    print_every = 1e6
    n_walkers = 768 if problem_difficulty == "hard" else 256

    swarm = MathySwarm(
        model=lambda env: DiscreteMasked(env=env),
        env=lambda: FragileMathyEnv(env=env),
        tree=HistoryTree,
        n_walkers=n_walkers,
        max_iters=max_iters,
        prune_tree=prune_tree,
        reward_scale=reward_scale,
        distance_scale=distance_scale,
        distance_function=mathy_dist,
        minimize=minimize,
    )

    if not use_vis:
        _ = swarm.run(print_every=print_every)
    else:
        holoviews.extension("bokeh")
        viz = SwarmViz1D(swarm, stream_interval=print_every)
        viz.plot()
        viz.run(print_every=print_every)

    best_ix = swarm.walkers.states.cum_rewards.argmax()
    best_id = swarm.walkers.states.id_walkers[best_ix]
    path = swarm.tree.get_branch(best_id, from_hash=True)
    env.render(last_action=-1, last_reward=0.0)
    for s, a in zip(path[0][1:], path[1]):
        _, _, r, _, info = env.step(state=s, action=a)
        env.render(last_action=a, last_reward=r)
        time.sleep(0.01)


for e_type in ["complex", "binomial", "poly", "poly-blockers"]:
    for e_difficulty in ["easy", "normal", "hard"]:
        solve_problem(e_type, e_difficulty)
