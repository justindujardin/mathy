import copy
import sys
import time
import traceback

import gym
import holoviews
import numpy as np
import srsly
from gym import spaces

from bokeh.plotting import show
from fragile.core.dt_sampler import GaussianDt
from fragile.core.env import DiscreteEnv
from fragile.core.models import DiscreteUniform, Bounds
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.swarm import Swarm
from fragile.core.tree import HistoryTree
from fragile.core.utils import StateDict
from fragile.core.walkers import Walkers
from fragile.dataviz import AtariViz, LandscapeViz, Summary, SwarmViz, SwarmViz1D
from mathy import MathTypeKeysMax, MathyEnvState, is_terminal_transition
from mathy.envs.gym import MathyGymEnv
from plangym import ParallelEnvironment
from plangym.env import Environment
import os


# Print explored mathy states when True
verbose = False
use_mp = True
dt = GaussianDt(min_dt=1, max_dt=1, loc_dt=4, scale_dt=2)
prune_tree = True
max_iters = 100
reward_scale = 5
distance_scale = 10
minimize = False
print_every = 10
use_vis = False
environment = "poly"
difficulty = "easy"
# the hard difficulty problems tend to have >= 100 nodes and need more workers to find
# solutions in such a large space since the workers can't go back in time.
n_walkers = 768 if difficulty == "hard" else 256


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
        n_repeat_actions = model_states.dt if hasattr(model_states, "dt") else 1
        new_states, observs, rewards, ends, infos = self._env.step_batch(
            actions=actions, states=env_states.states, n_repeat_action=n_repeat_actions
        )
        game_ends = [inf["done"] for inf in infos]
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
        self, name: str, n_repeat_action: int = 1, wrappers=None, **kwargs,
    ):
        self._env_kwargs = kwargs
        super(FragileEnvironment, self).__init__(
            name=name, n_repeat_action=n_repeat_action
        )
        self._env: MathyGymEnv = gym.make(
            f"mathy-{environment}-{difficulty}-v0", verbose=verbose
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
        # if reward > 0.0:
        #     print(f"r = {reward}")
        terminal = info.get("done", False)
        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, terminal, info
        return obs, reward, terminal, info

    def step_batch(
        self, actions, states=None, n_repeat_action: [int, np.ndarray] = None
    ) -> tuple:
        n_repeat_action = (
            n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        )
        n_repeat_action = (
            n_repeat_action.astype("i")
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        data = [
            self.step(action, state, n_repeat_action=dt)
            for action, state, dt in zip(actions, states, n_repeat_action)
        ]
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


if use_mp:
    env = ParallelEnvironment(
        env_class=FragileEnvironment,
        name="arc_v0",
        clone_seeds=True,
        autoreset=True,
        blocking=False,
    )
else:
    env = FragileEnvironment(name="arc_v0")


def arc_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=1)


swarm = Swarm(
    model=lambda env: DiscreteUniform(env=env, critic=dt),
    env=lambda: FragileMathyEnv(env=env),
    tree=HistoryTree,
    n_walkers=n_walkers,
    max_iters=max_iters,
    prune_tree=prune_tree,
    reward_scale=reward_scale,
    distance_scale=distance_scale,
    distance_function=arc_dist,
    minimize=minimize,
)

if not use_vis:
    _ = swarm.run(print_every=print_every)
else:
    # TODO: I don't know how bokeh/holoviews work outside of notebooks |x_X|
    holoviews.extension("bokeh")
    viz = SwarmViz1D(swarm, stream_interval=print_every)
    # plot = viz.plot()
    # show(plot)
    viz.run(print_every=print_every)


best_ix = swarm.walkers.states.cum_rewards.argmax()
best_id = swarm.walkers.states.id_walkers[best_ix]
path = swarm.tree.get_branch(best_id, from_hash=True)

env.render(last_action=-1, last_reward=0.0)
last_state = path[0][-1]
env_state = MathyEnvState.from_np(last_state)

# TODO: Terminal rewards and states are not included in the final path! :(
last_transition = env.mathy.get_state_transition(env_state)
solved = "SOLVED" if is_terminal_transition(last_transition) else "FAILED"
# TODO: I'm not sure this way of skipping invalid moves is correct...
for s, a in zip(path[0][1:], path[1]):
    _, _, r, _, info = env.step(state=s, action=a)
    if info.get("valid", False) is True:
        env.render(last_action=a, last_reward=r)
    time.sleep(0.05)
print(f"{solved} problem!")
print(f"Best reward: {swarm.walkers.states.best_reward}")
print("Agent History:")
print("\n".join([h.raw for h in env_state.agent.history]))
