import asyncio
from typing import Callable, Dict, List, Tuple

import numpy

from fragile.core.env import Environment as CoreEnv
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import split_args_in_chunks, split_kwargs_in_chunks, StateDict
from fragile.distributed.ray import ray

# The type hints of the base class are not supported by cloudpickle
# and will raise errors in Python3.6


@ray.remote
class Environment:
    """
    :class:`fragile.Environment` remote interface to be used with ray.

    Wraps a :class:`fragile.Environment` passed as a callable.
    """

    def __init__(self, env_callable: Callable, env_kwargs: dict = None):
        """
        Initialize a :class:`Environment`.

        Args:
            env_callable: Callable that returns a :class:`fragile.Environment`.
            env_kwargs: Passed to ``env_callable``.

        """
        env_kwargs = {} if env_kwargs is None else env_kwargs
        self.env = env_callable(**env_kwargs)

    def get(self, name: str, default=None):
        """
        Get an attribute from the wrapped environment.

        Args:
            name: Name of the target attribute.

        Returns:
            Attribute from the wrapped :class:`fragile.Environment`.

        """
        try:
            return getattr(self.env, name)
        except Exception:
            return default

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Step the wrapped :class:`fragile.Environment`.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            States representing the next state of the environment and all \
            the needed information.

        """
        return self.env.step(model_states=model_states, env_states=env_states)

    def states_from_data(self, batch_size: int, kwargs) -> StatesEnv:
        """
        Initialize a :class:`StatesEnv` with the data provided as kwargs.

        Args:
            batch_size: Number of elements in the first dimension of the \
                       :class:`State` attributes.
            **kwargs: Attributes that will be added to the returned :class:`States`.

        Returns:
            A new :class:`StatesEmv` created with the ``params_dict``, and \
            updated with the attributes passed as keyword arguments.

        """
        return self.env.states_from_data(batch_size=batch_size, **kwargs)

    def make_transitions(self, *args, **kwargs) -> Dict:
        """
        Return the data corresponding to the new state of the environment after \
        using the input data to make the corresponding state transition.

        Args:
            *args: List of arguments passed if the returned value from the \
                  ``states_to_data`` function of the class was a tuple.
            **kwargs: Keyword arguments passed if the returned value from the \
                  ``states_to_data`` function of the class was a dictionary.

        Returns:
            Dictionary containing the data representing the state of the environment \
            after the state transition. The keys of the dictionary are the names of \
            the data attributes and its values are arrays representing a batch of \
            new values for that attribute.

            The :class:`StatesEnv` returned by ``step`` will contain the returned \
            data.

        """
        return self.env.make_transitions(*args, **kwargs)

    def states_to_data(self, model_states: StatesModel, env_states: StatesEnv):
        """
        Extract the data from the :class:`StatesEnv` and the :class:`StatesModel` \
        and return the values that will be passed to ``make_transitions``.

        Args:
            model_states: :class:`StatesModel` representing the data to be used \
                         to act on the environment.
            env_states: :class:`StatesEnv` representing the data to be set in \
                       the environment.

        Returns:
            Tuple of arrays or dictionary of arrays. If the returned value is a \
            tuple it will be passed as *args to ``make_transitions``. If the returned \
            value is a dictionary it will be passed as **kwargs to ``make_transitions``.

        """
        return self.env.states_to_data(model_states=model_states, env_states=env_states)

    def reset(self, batch_size: int = 1, env_states: StatesEnv = None, **kwargs) -> StatesEnv:
        """
        Reset the wrapped :class:`fragile.Environment` and return an States class \
        with batch_size copies of the initial state.

        Args:
           batch_size: Number of walkers that the resulting state will have.
           env_states: States class used to set the environment to an arbitrary \
                       state.
           kwargs: Additional keyword arguments not related to environment data.

        Returns:
           States class containing the information of the environment after the \
            reset.

        """
        return self.env.reset(batch_size=batch_size, env_states=env_states, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary of the wrapped :class:`fragile.Environment`."""
        return self.env.get_params_dict()


@ray.remote
def merge_data(data_dicts: List[Dict[str, numpy.ndarray]]):
    """
    Group together the data returned from calling ``make_transitions`` in several \
    remote :class:`Environment`.
    """

    def group_data(vals):
        try:
            return (
                numpy.vstack(vals) if len(vals[0].shape) > 1 else numpy.concatenate(vals).flatten()
            )
        except Exception:
            raise ValueError("MIAU: %s %s" % (len(vals), vals[0].shape))

    kwargs = {}
    for k in data_dicts[0].keys():
        grouped = group_data([ddict[k] for ddict in data_dicts])
        kwargs[k] = grouped
    return kwargs


@ray.remote
class RayEnv(CoreEnv):
    """Step an :class:`Environment` in parallel using ``ray``."""

    def __init__(
        self, env_callable: Callable, n_workers: int, env_kwargs: dict = None,
    ):
        """
        Initialize a :class:`RayEnv`.

        Args:
            env_callable: Returns the :class:`Environment` that will be distributed.
            n_workers: Number of processes that will step the \
                       :class:`Environment` in parallel.
            env_kwargs: Passed to ``env_callable``.

        """
        env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_workers = n_workers
        self.envs = [
            Environment.remote(env_callable=env_callable, env_kwargs=env_kwargs)
            for _ in range(n_workers)
        ]
        self._remote_env = Environment.remote(env_callable=env_callable, env_kwargs=env_kwargs)
        self._local_env = env_callable(**env_kwargs)
        CoreEnv.__init__(
            self,
            states_shape=self._local_env.states_shape,
            observs_shape=self._local_env.observs_shape,
        )

    def get(self, name: str, default=None):
        """
        Get an attribute from the wrapped environment.

        Args:
            name: Name of the target attribute.

        Returns:
            Attribute from the wrapped :class:`fragile.Environment`.

        """
        try:
            return self._local_env.__getattribute_(name)
        except Exception:
            return default

    async def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Set the environment to the target states by applying the specified \
        actions an arbitrary number of time steps.

        The state transitions will be calculated in parallel.

        Args:
            model_states: :class:`StatesModel` representing the data to be used \
                         to act on the environment.
            env_states: :class:`StatesEnv` representing the data to be set in \
                        the environment.

        Returns:
            :class:`StatesEnv` containing the information that describes the \
            new state of the Environment.

        """
        transition_data_id = self._remote_env.states_to_data.remote(
            model_states=model_states, env_states=env_states
        )
        """if not isinstance(transition_data, (dict, tuple)):
            raise ValueError(
                "The returned values from states_to_data need to "
                "be an instance of dict or tuple. "
                "Got %s instead" % type(transition_data)
            )"""
        transition_data = await transition_data_id
        new_data_promise = (
            self.make_transitions(*transition_data)
            if isinstance(transition_data, tuple)
            else self.make_transitions(**transition_data)
        )
        new_data = await new_data_promise
        new_env_state = await self._remote_env.states_from_data.remote(len(env_states), new_data)
        return new_env_state

    async def make_transitions(self, *args, **kwargs):
        """
        Forward the make_transitions arguments to the parallel environments \
        splitting them in batches of similar size.
        """
        chunk_data = self._split_inputs_in_chunks(*args, **kwargs)
        split_results = await asyncio.gather(*self._make_transitions(chunk_data))
        merged = merge_data.remote(split_results)
        return merged

    def _split_inputs_in_chunks(self, *args, **kwargs):
        self.kwargs_mode = len(args) == 0
        if self.kwargs_mode:

            return split_kwargs_in_chunks(kwargs, len(self.envs))
        else:
            return split_args_in_chunks(args, len(self.envs))

    def _make_transitions(self, split_results):
        results_ids = [
            env.make_transitions.remote(**chunk)
            if self.kwargs_mode
            else env.make_transitions.remote(*chunk)
            for env, chunk in zip(self.envs, split_results)
        ]
        return results_ids

    async def reset(
        self, batch_size: int = 1, env_states: StatesEnv = None, *args, **kwargs
    ) -> Tuple:
        """
        Reset the environment to the start of a new episode and returns a new \
        States instance describing the state of the Environment.

        Args:
            batch_size: Number of walkers that the returned state will have.
            env_states: :class:`StatesEnv` representing the data to be set in \
                        the environment.
            *args: Passed to the internal environment ``reset``.
            **kwargs: Passed to the internal environment ``reset``.

        Returns:
            States instance describing the state of the Environment. The first \
            dimension of the data tensors (number of walkers) will be equal to \
            batch_size.

        """
        reset_ids = [
            env.reset.remote(batch_size=batch_size, env_states=env_states, *args, **kwargs)
            for env in self.envs
        ]
        await asyncio.gather(*reset_ids)
        env_state = self._local_env.reset(
            batch_size=batch_size, env_states=env_states, *args, **kwargs
        )
        return env_state
