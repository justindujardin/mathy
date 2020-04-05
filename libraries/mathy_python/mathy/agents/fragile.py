"""Use Fractal Monte Carlo search in order to solve mathy problems without a
trained neural network."""

import copy
import time
from typing import Dict, List, Optional, Union

import gym
import numpy as np
from fragile.core.env import DiscreteEnv
from fragile.core.models import DiscreteModel
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.swarm import Swarm
from fragile.core.utils import StateDict
from fragile.core.walkers import Walkers
from gym import spaces
from pydantic import BaseModel
from wasabi import msg

from .. import (
    EnvRewards,
    MathTypeKeysMax,
    MathyEnv,
    MathyEnvState,
    about,
    is_terminal_transition,
)
from ..envs.gym import MathyGymEnv


class SwarmConfig(BaseModel):
    verbose: bool = False
    n_walkers: int = 512
    max_iters: int = 100


def mathy_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=1)


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
            actions=actions, model_states=model_states, batch_size=batch_size, **kwargs,
        )


class FragileMathyEnv(DiscreteEnv):
    """The DiscreteEnv acts as an interface with `plangym` discrete actions.

    It can interact with any environment that accepts discrete actions and \
    follows the interface of `plangym`.
    """

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "normal",
        problem: str = None,
        max_steps: int = 64,
        **kwargs,
    ):
        self._env = FragileEnvironment(
            name=name,
            environment=environment,
            difficulty=difficulty,
            problem=problem,
            max_steps=max_steps,
            **kwargs,
        )
        self._n_actions = self._env.action_space.n
        super(DiscreteEnv, self).__init__(
            states_shape=self._env.get_state().shape,
            observs_shape=self._env.observation_space.shape,
        )

    def __getattr__(self, item):
        return getattr(self._env, item)

    def make_transitions(
        self, states: np.ndarray, actions: np.ndarray, dt: Union[np.ndarray, int]
    ) -> Dict[str, np.ndarray]:
        """
        Step the underlying :class:`plangym.Environment` using the ``step_batch`` \
        method of the ``plangym`` interface.
        """
        new_states, observs, rewards, oobs, infos = self._env.step_batch(
            actions=actions, states=states
        )
        terminals = [inf.get("done", False) for inf in infos]
        data = {
            "states": np.array(new_states),
            "observs": np.array(observs),
            "rewards": np.array(rewards),
            "oobs": np.array(oobs),
            "terminals": np.array(terminals),
        }
        return data


class FragileEnvironment:
    """Fragile Environment for solving Mathy problems."""

    problem: Optional[str]

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "normal",
        problem: str = None,
        max_steps: int = 64,
        **kwargs,
    ):
        self._env: MathyGymEnv = gym.make(
            f"mathy-{environment}-{difficulty}-v0",
            np_observation=True,
            error_invalid=False,
            env_problem=problem,
            **kwargs,
        )
        self.observation_space = spaces.Box(
            low=0, high=MathTypeKeysMax, shape=(256, 256, 1), dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(self._env.action_size)
        self.problem = problem
        self.max_steps = max_steps
        self._env.reset()

    def get_state(self) -> np.ndarray:
        assert self._env.state is not None, "env required to get_state"
        return self._env.state.to_np()

    def set_state(self, state: np.ndarray):
        assert self._env is not None, "env required to set_state"
        self._env.state = MathyEnvState.from_np(state)
        return state

    def step(self, action: int, state: np.ndarray = None,) -> tuple:
        assert self._env is not None, "env required to step"
        assert state is not None, "only works with state stepping"
        self.set_state(state)
        obs, reward, _, info = self._env.step(action)
        oob = not info.get("valid", False)
        new_state = self.get_state()
        return new_state, obs, reward, oob, info

    def step_batch(
        self, actions, states=None, n_repeat_action: Union[int, np.ndarray] = None
    ) -> tuple:
        data = [self.step(action, state) for action, state in zip(actions, states)]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            new_state, obs, _reward, end, info = d
            new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        return new_states, observs, rewards, terminals, infos

    def reset(self, batch_size: int = 1):
        assert self._env is not None, "env required to reset"
        obs = self._env.reset()
        return self.get_state(), obs


def swarm_solve(problem: str, config: SwarmConfig):
    env_callable = lambda: FragileMathyEnv(
        name="mathy_v0", problem=problem, repeat_problem=True
    )
    mathy_env: MathyEnv = env_callable()._env._env.mathy
    swarm = Swarm(
        model=lambda env: DiscreteMasked(env=env),
        env=env_callable,
        reward_limit=EnvRewards.WIN,
        n_walkers=config.n_walkers,
        max_epochs=config.max_iters,
        reward_scale=1,
        distance_scale=3,
        distance_function=mathy_dist,
        show_pbar=False,
    )

    with msg.loading(f"Solving {problem} ..."):
        _ = swarm.run()

    if swarm.walkers.best_reward > EnvRewards.WIN:
        last_state = MathyEnvState.from_np(swarm.walkers.states.best_state)
        msg.good(f"Solved! {problem} = {last_state.agent.problem}")
        mathy_env.print_history(last_state)
    else:
        msg.fail(f"Failed to find a solution :(")
