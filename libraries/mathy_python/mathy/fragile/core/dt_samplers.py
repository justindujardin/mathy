from typing import Optional

import numpy as np

from fragile.core.base_classes import BaseCritic
from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import float_type, StateDict


class BaseDtSampler(BaseCritic):
    """
    :class:`Critic` that returns contains the attribute ``dt`` in its :class:`States`.

    The ``dt`` value represents the number of time steps a given action will \
    be applied. The returned values are floats so the critic can also be used \
    to sample a learning rate. If you need discrete values transform the sampled \
    array to integers in the :class:`Environment`, or change the data type \
    overriding the default ``get_params_dict`` behavior.

    The ``dt`` value will also be replicated in the ``critic_score`` attribute \
    of the states to comply with the critic interface.
    """

    def __init__(self, discrete_values: bool = True):
        """
        Initialize a :class:`BaseDtSampler`.

        Args:
            discrete_values: If ``True`` return discrete time step values. If \
                            ``False`` allow to return floating point time steps.
        """
        self._dtype = float_type if not discrete_values else int
        super(BaseDtSampler, self).__init__()

    def get_params_dict(self) -> StateDict:
        """Return the dictionary with the parameters to create a new `GaussianDt` critic."""
        base_params = super(BaseDtSampler, self).get_params_dict()
        params = {"dt": {"dtype": self._dtype}}
        base_params.update(params)
        return base_params

    def calculate(
        self,
        batch_size: Optional[int] = None,
        model_states: Optional[StatesModel] = None,
        env_states: Optional[StatesEnv] = None,
        walkers_states: Optional[StatesWalkers] = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled time step.

        """
        raise NotImplementedError


class ConstantDt(BaseDtSampler):
    """Apply the actions sampled a constant number of time steps."""

    def __init__(self, dt: float, discrete_values: bool = False):
        """
        Initialize a :class:`ConstantDt`.

        Args:
            dt: Number of time steps that each action will be applied.
            discrete_values: If ``True`` return discrete time step values. If \
                            ``False`` allow to return floating point time steps.
        """
        self.dt = dt
        super(ConstantDt, self).__init__(discrete_values=discrete_values)

    def calculate(
        self,
        batch_size: Optional[int] = None,
        model_states: Optional[StatesModel] = None,
        env_states: Optional[StatesEnv] = None,
        walkers_states: Optional[StatesWalkers] = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled time step.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        dt = np.ones(batch_size, dtype=self._dtype) * self.dt
        states = self.states_from_data(batch_size=batch_size, critic_score=dt, dt=dt)
        return states


class UniformDt(BaseDtSampler):
    """Sample the ``dt`` values for each action from a discrete uniform distribution."""

    def __init__(self, min_dt: float = 1.0, max_dt: float = 1.0, discrete_values: bool = False):
        """
        Initialize a :class:`GaussianDt`.

        Args:
            min_dt: Minimum dt that can be sampled.
            max_dt: Maximum dt that can be sampled.
            discrete_values: If ``True`` return discrete time step values. If \
                            ``False`` allow to return floating point time steps.

        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        super(UniformDt, self).__init__(discrete_values=discrete_values)

    def calculate(
        self,
        batch_size: Optional[int] = None,
        model_states: Optional[StatesModel] = None,
        env_states: Optional[StatesEnv] = None,
        walkers_states: Optional[StatesWalkers] = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled time step for each value drawn for a \
            uniform distribution.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        dt = self.random_state.uniform(low=self.min_dt, high=self.max_dt, size=batch_size)
        dt = dt.astype(dtype=self._dtype)
        states = self.states_from_data(batch_size=batch_size, critic_score=dt, dt=dt)
        return states


class GaussianDt(BaseDtSampler):
    """
    Sample an additional vector of clipped gaussian random variables, and \
    stores it in an attribute called `dt`.
    """

    def __init__(
        self,
        min_dt: float = 1.0,
        max_dt: float = 1.0,
        loc_dt: float = 0.0,
        scale_dt: float = 1.0,
        discrete_values: bool = False,
    ):
        """
        Initialize a :class:`GaussianDt`.

        Args:
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.
            discrete_values: If ``True`` return discrete time step values. If \
                            ``False`` allow to return floating point time steps.

        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = loc_dt
        self.std_dt = scale_dt
        super(GaussianDt, self).__init__(discrete_values=discrete_values)

    def calculate(
        self,
        batch_size: Optional[int] = None,
        model_states: Optional[StatesModel] = None,
        env_states: Optional[StatesEnv] = None,
        walkers_states: Optional[StatesWalkers] = None,
    ) -> States:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled time step values drawn from a gaussian \
            distribution.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        dt = self.random_state.normal(loc=self.mean_dt, scale=self.std_dt, size=batch_size)
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(self._dtype)
        states = self.states_from_data(batch_size=batch_size, critic_score=dt, dt=dt)
        return states
