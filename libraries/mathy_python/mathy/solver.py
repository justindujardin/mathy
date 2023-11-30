"""Use Fractal Monte Carlo search in order to solve mathy problems without a
trained neural network."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from mathy_core import MathTypeKeysMax
from mathy_envs import EnvRewards, MathyEnv, MathyEnvState
from mathy_envs.gym import MathyGymEnv
from wasabi import msg

from .fragile.core.env import DiscreteEnv
from .fragile.core.models import DiscreteModel
from .fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from .fragile.core.swarm import Swarm
from .fragile.core.distributed_env import ParallelEnv


@dataclass
class SwarmConfig:
    use_mp: bool = True
    history: bool = False
    history_names: List[str] = field(
        default_factory=lambda: ["states", "actions", "rewards"]
    )
    single_problem: bool = False
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
            masks = env_states.observs[:, -self.n_actions :]
            actions = random_choice_prob_index(masks)
        else:
            actions = self.random_state.randint(0, self.n_actions, size=batch_size)
        return self.update_states_with_critic(
            actions=actions,
            model_states=model_states,
            batch_size=batch_size,
            **kwargs,
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
        difficulty: str = "easy",
        problem: Optional[str] = None,
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

    @property
    def unwrapped(self) -> MathyGymEnv:
        return cast(MathyGymEnv, self._env.unwrapped)

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "normal",
        problem: Optional[str] = None,
        max_steps: int = 64,
        **kwargs,
    ):
        import gymnasium as gym
        from gymnasium import spaces
        from mathy_envs.gym import MathyGymEnv

        self._env = gym.make(
            f"mathy-{environment}-{difficulty}-v0",
            invalid_action_response="terminal",
            env_problem=problem,
            mask_as_probabilities=True,
            **kwargs,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=MathTypeKeysMax,
            shape=(256, 256, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(self._env.unwrapped.action_size)
        self.problem = problem
        self.max_steps = max_steps
        self._env.reset()

    def get_state(self) -> np.ndarray:
        assert self.unwrapped.state is not None, "env required to get_state"
        return self.unwrapped.state.to_np(2048)

    def set_state(self, state: np.ndarray):
        assert self.unwrapped is not None, "env required to set_state"
        self.unwrapped.state = MathyEnvState.from_np(state)
        return state

    def step(
        self, action: int, state: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Any, bool, Dict[str, object]]:
        assert self._env is not None, "env required to step"
        assert state is not None, "only works with state stepping"
        self.set_state(state)
        obs, reward, _, _, info = self._env.step(action)
        oob = not info.get("valid", False)
        new_state = self.get_state()
        return new_state, obs, reward, oob, info

    def step_batch(
        self,
        actions,
        states: Optional[Any] = None,
        n_repeat_action: Optional[Union[int, np.ndarray]] = None,
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
        obs, info = self._env.reset()
        return self.get_state(), obs


def mathy_swarm(config: SwarmConfig, env_callable=None) -> Swarm:
    if env_callable is None:
        env_callable = lambda: FragileMathyEnv(
            name="mathy_v0", repeat_problem=config.single_problem
        )
    if config.use_mp:
        env_callable = ParallelEnv(env_callable=env_callable)
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
    return swarm


def swarm_solve(
    problems: Union[List[str], str],
    config: SwarmConfig,
    max_steps: Union[List[int], int] = 256,
    silent: bool = False,
) -> Swarm:
    single_problem: bool = isinstance(problems, str)
    if single_problem:
        problems = [problems]
    if isinstance(max_steps, int):
        max_steps = [max_steps] if single_problem else [max_steps] * len(problems)
    assert len(problems) > 0, "no problems to solve"
    assert len(problems) == len(max_steps)
    assert isinstance(problems, list)
    current_problem: str = problems.pop(0)
    current_max_moves: int = max_steps.pop(0)

    def env_callable():
        nonlocal current_problem, current_max_moves
        return FragileMathyEnv(
            name="mathy_v0",
            problem=current_problem,
            repeat_problem=True,
            max_steps=current_max_moves,
        )

    mathy_env: MathyEnv = env_callable()._env.unwrapped.mathy
    swarm: Swarm = mathy_swarm(config, env_callable)
    while True:
        if not silent:
            with msg.loading(f"Solving {current_problem} ..."):
                swarm.run()
        else:
            swarm.run()

        if not silent:
            if swarm.walkers.best_reward > EnvRewards.WIN:
                last_state = MathyEnvState.from_np(swarm.walkers.states.best_state)
                msg.good(f"Solved! {current_problem} = {last_state.agent.problem}")
                mathy_env.print_history(last_state)
            else:
                msg.fail(f"Failed to find a solution :(")

        if len(max_steps) > 0:
            current_max_moves = max_steps.pop(0)
            current_problem = problems.pop(0)
        else:
            break
    return swarm
