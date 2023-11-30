from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy

from . import BaseWrapper, Environment, StatesEnv, StatesModel
from .utils import StateDict


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

    def reset(
        self, batch_size: int = 1, env_states: StatesEnv = None, **kwargs
    ) -> StatesEnv:
        return self.unwrapped.__class__.reset(
            self.unwrapped, batch_size=batch_size, env_states=env_states, **kwargs
        )

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        return self.unwrapped.__class__.step(
            self.unwrapped, model_states=model_states, env_states=env_states
        )

    def make_transitions(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        return self.unwrapped.__class__.make_transitions(
            self.unwrapped, *args, **kwargs
        )

    def states_to_data(
        self, model_states: StatesModel, env_states: StatesEnv
    ) -> Union[Dict[str, numpy.ndarray], Tuple[numpy.ndarray, ...]]:
        return self.unwrapped.__class__.states_to_data(
            self.unwrapped, model_states=model_states, env_states=env_states
        )
