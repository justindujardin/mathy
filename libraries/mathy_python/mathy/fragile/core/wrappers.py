from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy

from fragile.core import (
    BaseCritic,
    BaseWrapper,
    Environment,
    Model,
    OneWalker,
    States,
    StatesEnv,
    StatesModel,
    StatesWalkers,
    Swarm,
    Walkers,
)
from fragile.core.base_classes import BaseTree
from fragile.core.utils import StateDict


class CriticWrapper(BaseWrapper, BaseCritic):
    def __init__(self, critic: BaseCritic, name: str = "_critic"):
        BaseWrapper.__init__(self, critic, name=name)

    def get_params_dict(self) -> StateDict:
        return self.unwrapped.__class__.get_params_dict(self.unwrapped)

    def calculate(
        self,
        batch_size: int = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs,
    ) -> States:
        return self.unwrapped.__class__.calculate(
            self.unwrapped,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )

    def reset(
        self,
        batch_size: int = 1,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs
    ) -> Union[States, None]:
        return self.unwrapped.__class__.reset(
            self.unwrapped,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )

    def update(
        self,
        batch_size: int = 1,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs
    ) -> Union[States, None]:
        return self.unwrapped.__class__.update(
            self.unwrapped,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )


class ModelWrapper(BaseWrapper, Model):
    def __init__(self, model: Model, name: str = "_model"):
        BaseWrapper.__init__(self, model, name=name)

    def get_params_dict(self) -> StateDict:
        return self.unwrapped.__class__.get_params_dict(self.unwrapped)

    def sample(
        self,
        batch_size: int,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs
    ) -> StatesModel:
        return self.unwrapped.__class__.sample(
            self.unwrapped,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )

    def predict(
        self,
        batch_size: int = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs,
    ) -> StatesModel:
        return self.unwrapped.__class__.predict(
            self.unwrapped,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )

    def reset(
        self, batch_size: int = 1, model_states: StatesModel = None, *args, **kwargs
    ) -> StatesModel:
        return self.unwrapped.__class_.reset(
            self.unwrapped, batch_size=batch_size, model_states=model_states, *args, **kwargs
        )

    def add_critic_params(
        self, params: dict, override_params: bool = True, *args, **kwargs
    ) -> StateDict:
        return self.unwrapped.__class__.add_critic_params(
            self.unwrapped, params=params, override_params=override_params, *args, **kwargs
        )

    def update_states_with_critic(
        self, actions: numpy.ndarray, batch_size: int, model_states: StatesModel, **kwargs
    ) -> StatesModel:
        return self.unwrapped.__class__.update_states_with_critic(
            self.unwrapped,
            actions=actions,
            batch_size=batch_size,
            model_states=model_states,
            **kwargs
        )


class EnvWrapper(BaseWrapper, Environment):
    def __init__(self, env: Environment, name: str = "_env"):
        BaseWrapper.__init__(self, env, name=name)

    def get_params_dict(self) -> StateDict:
        return self.unwrapped.__class__.get_params_dict(self.unwrapped)

    def states_from_data(
        self, batch_size, states, observs, rewards, oobs, terminals=None, **kwargs
    ) -> StatesEnv:
        return self.unwrapped.__class__.states_from_data(
            self.unwrapped,
            batch_size=batch_size,
            states=states,
            observs=observs,
            rewards=rewards,
            oobs=oobs,
            terminals=terminals,
            **kwargs
        )

    def reset(self, batch_size: int = 1, env_states: StatesEnv = None, **kwargs) -> StatesEnv:
        return self.unwrapped.__class__.reset(
            self.unwrapped, batch_size=batch_size, env_states=env_states, **kwargs
        )

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        return self.unwrapped.__class__.step(
            self.unwrapped, model_states=model_states, env_states=env_states
        )

    def make_transitions(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        return self.unwrapped.__class__.make_transitions(self.unwrapped, *args, **kwargs)

    def states_to_data(
        self, model_states: StatesModel, env_states: StatesEnv
    ) -> Union[Dict[str, numpy.ndarray], Tuple[numpy.ndarray, ...]]:
        return self.unwrapped.__class__.states_to_data(
            self.unwrapped, model_states=model_states, env_states=env_states
        )


class WalkersWrapper(BaseWrapper, Walkers):
    def __init__(self, walkers: Walkers, name: str = "_walkers"):
        BaseWrapper.__init__(self, walkers, name=name)

    def __repr__(self):
        return self.unwrapped.__class__.__repr__(self.unwrapped)

    def _print_stats(self) -> str:
        return self.unwrapped.__class__._print_stats(self.unwrapped)

    def ids(self) -> List[int]:
        return self.unwrapped.__class__.ids(self.unwrapped)

    def update_ids(self):
        return self.unwrapped.__class__.update_ids(self.unwrapped)

    def calculate_end_condition(self) -> bool:
        return self.unwrapped.__class__.calculate_end_condition(self.unwrapped)

    def calculate_distances(self) -> None:
        return self.unwrapped.__class__.calculate_distances(self.unwrapped)

    def calculate_virtual_reward(self):
        return self.unwrapped.__class__.calculate_virtual_reward(self.unwrapped)

    def get_in_bounds_compas(self) -> numpy.ndarray:
        return self.unwrapped.__class__.get_in_bounds_compas(self.unwrapped)

    def update_clone_probs(self) -> None:
        return self.unwrapped.__class__.update_clone_probs(self.unwrapped)

    def balance(self):
        return self.unwrapped.__class__.balance(self.unwrapped)

    def clone_walkers(self) -> None:
        return self.unwrapped.__class__.clone_walkers(self.unwrapped)

    def reset(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
        *args,
        **kwargs,
    ):
        return self.unwrapped.__class__.reset(
            self.unwrapped,
            env_states=env_states,
            model_states=model_states,
            walkers_states=walkers_states,
            *args,
            **kwargs
        )

    def update_states(
        self, env_states: StatesEnv = None, model_states: StatesModel = None, **kwargs
    ):
        return self.unwrapped.__class__.update_states(
            self.unwrapped, env_states=env_states, model_states=model_states, **kwargs
        )

    def _accumulate_and_update_rewards(self, rewards: numpy.ndarray):
        return self.unwrapped.__class__._accumulate_and_update_rewards(
            self.unwrapped, rewards=rewards
        )

    def fix_best(self):
        return self.unwrapped.__class__.fix_best(self.unwrapped)

    def get_best_index(self) -> int:
        return self.unwrapped.__class__.get_best_index(self.unwrapped)

    def update_best(self):
        return self.unwrapped.__class__.update_best(self.unwrapped)


class SwarmWrapper(BaseWrapper, Swarm):
    def __init__(self, swarm, name: str = "_swarm"):
        BaseWrapper.__init__(self, swarm, name=name)

    def __len__(self):
        return self.unwrapped.__class__.__len__(self.unwrapped)

    def __repr__(self):
        return self.unwrapped.__class__.__repr__(self.unwrapped)

    def get(self, name: str, default: Any = None) -> Any:
        return self.unwrapped.__class__.get(self.unwrapped, name, default)

    def increment_epoch(self) -> None:
        return self.unwrapped.__class__.increment_epoch(self.unwrapped)

    def reset(
        self,
        root_walker: OneWalker = None,
        walkers_states: StatesWalkers = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        **kwargs,
    ):
        return self.unwrapped.__class__.reset(
            self.unwrapped,
            root_walker,
            walkers_states=walkers_states,
            model_states=model_states,
            env_states=env_states,
            **kwargs
        )

    def run(
        self,
        root_walker: OneWalker = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        report_interval: int = None,
        show_pbar: bool = None,
    ):
        return self.unwrapped.__class__.run(
            self.unwrapped,
            root_walker=root_walker,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            report_interval=report_interval,
            show_pbar=show_pbar,
        )

    def step_walkers(self) -> None:
        return self.unwrapped.__class__.step_walkers(self.unwrapped)

    def init_swarm(
        self,
        env_callable: Callable[[], Environment],
        model_callable: Callable[[Environment], Model],
        walkers_callable: Callable[..., Walkers],
        n_walkers: int,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        tree: Callable[[], BaseTree] = None,
        prune_tree: bool = True,
        *args,
        **kwargs
    ):
        return self.unwrapped.__class__.init_swarm(
            self.unwrapped,
            env_callable=env_callable,
            model_callable=model_callable,
            walkers_callable=walkers_callable,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            distance_scale=distance_scale,
            tree=tree,
            prune_tree=prune_tree,
            *args,
            **kwargs,
        )

    def get_run_loop(self, show_pbar: bool = None) -> Iterable[int]:
        return self.unwrapped.__class__.get_run_loop(self.unwrapped, show_pbar=show_pbar)

    def setup_notebook_container(self):
        return self.unwrapped.__class__.setup_notebook_container(self.unwrapped)

    def report_progress(self):
        return self.unwrapped.__class__.report_progress(self.unwrapped)

    def calculate_end_condition(self) -> bool:
        return self.unwrapped.__class__.calculate_end_condition(self.unwrapped)

    def step_and_update_best(self) -> None:
        return self.unwrapped.__class__.step_and_update_best(self.unwrapped)

    def balance_and_prune(self) -> None:
        return self.unwrapped.__class__.balance_and_prune(self.unwrapped)

    def run_step(self) -> None:
        return self.unwrapped.__class__.run_step(self.unwrapped)

    def update_tree(self, states_ids: List[int]) -> None:
        return self.unwrapped.__class__.update_tree(self.unwrapped, states_ids=states_ids)

    def prune_tree(self) -> None:
        return self.unwrapped.__class__.prune_tree(self.unwrapped)

    def _update_env_with_root(self, root_walker, env_states) -> StatesEnv:
        return self.unwrapped.__class__._update_env_with_root(
            self.unwrapped, root_walker=root_walker, env_states=env_states
        )


class TreeWrapper(BaseWrapper, BaseTree):
    def __init__(self, tree: BaseTree, name: str = "_tree"):
        BaseWrapper.__init__(self, tree, name=name)

    def add_states(
        self,
        parent_ids: List[int],
        env_states: States = None,
        model_states: States = None,
        walkers_states: States = None,
        n_iter: int = None,
    ) -> None:
        return self.unwrapped.__class__.add_states(
            self.unwrapped,
            parent_ids=parent_ids,
            env_states=env_states,
            model_states=model_states,
            walkers_states=walkers_states,
            n_iter=n_iter,
        )

    def reset(
        self,
        root_id: int = 0,
        root_hash: int = 0,
        env_states: States = None,
        model_states: States = None,
        walkers_states: States = None,
    ) -> None:
        return self.unwrapped.__class__.reset(
            self.unwrapped,
            root_id=root_id,
            root_hash=root_hash,
            env_states=env_states,
            model_states=model_states,
            walkers_states=walkers_states,
        )

    def prune_tree(self, alive_leafs: set, from_hash: bool = False) -> None:
        return self.unwrapped.__class__.prune_tree(
            self.unwrapped, alive_leafs=alive_leafs, from_hash=from_hash
        )
