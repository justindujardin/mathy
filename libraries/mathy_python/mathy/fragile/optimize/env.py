from typing import Callable, Dict, Tuple, Union

import numpy
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import minimize

from fragile.core.env import Environment
from fragile.core.models import Bounds
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import Scalar


class Function(Environment):
    """
    Environment that represents an arbitrary mathematical function bounded in a \
    given interval.
    """

    def __init__(
        self,
        function: Callable[[numpy.ndarray], numpy.ndarray],
        bounds: Bounds,
        custom_domain_check: Callable[[numpy.ndarray], numpy.ndarray] = None,
    ):
        """
        Initialize a :class:`Function`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      scalar. This function is applied to a batch of walker \
                      observations.
            bounds: :class:`Bounds` that defines the domain of the function.
            custom_domain_check: Callable that checks points inside the bounds \
                    to know if they are in a custom domain when it is not just \
                    a set of rectangular bounds. It takes a batch of points as \
                    input and returns an array of booleans. Each ``True`` value \
                    indicates that the corresponding point is **outside**  the \
                    ``custom_domain_check``.

        """
        if not isinstance(bounds, Bounds):
            raise TypeError("Bounds needs to be an instance of Bounds, found {}".format(bounds))
        self.function = function
        self.bounds = bounds
        self.custom_domain_check = custom_domain_check
        super(Function, self).__init__(observs_shape=self.shape, states_shape=self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the environment."""
        return self.bounds.shape

    @classmethod
    def from_bounds_params(
        cls,
        function: Callable,
        shape: tuple = None,
        high: Union[int, float, numpy.ndarray] = numpy.inf,
        low: Union[int, float, numpy.ndarray] = numpy.NINF,
        custom_domain_check: Callable[[numpy.ndarray], numpy.ndarray] = None,
    ) -> "Function":
        """
        Initialize a function defining its shape and bounds without using a :class:`Bounds`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      scalar. This function is applied to a batch of walker \
                      observations.
            shape: Input shape of the solution vector without taking into account \
                    the batch dimension. For example, a two dimensional function \
                    applied to a batch of 5 walkers will have shape=(2,), even though
                    the observations will have shape (5, 2)
            high: Upper bound of the function domain. If it's an scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the upper bound for each dimension.
            low: Lower bound of the function domain. If it's an scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the lower bound for each dimension.
            custom_domain_check: Callable that checks points inside the bounds \
                    to know if they are in a custom domain when it is not just \
                    a set of rectangular bounds.

        Returns:
            :class:`Function` with its :class:`Bounds` created from the provided arguments.

        """
        if (
            not isinstance(high, numpy.ndarray)
            and not isinstance(low, numpy.ndarray) is None
            and shape is None
        ):
            raise TypeError("Need to specify shape or high or low must be a numpy array.")
        bounds = Bounds(high=high, low=low, shape=shape)
        return Function(function=function, bounds=bounds, custom_domain_check=custom_domain_check)

    def __repr__(self):
        text = "{} with function {}, obs shape {},".format(
            self.__class__.__name__, self.function.__name__, self.shape,
        )
        return text

    def states_to_data(
        self, model_states: StatesModel, env_states: StatesEnv
    ) -> Dict[str, numpy.ndarray]:
        """
        Extract the data that will be used to make the state transitions.

        Args:
            model_states: :class:`StatesModel` representing the data to be used \
                         to act on the environment.
            env_states: :class:`StatesEnv` representing the data to be set in \
                       the environment.

        Returns:
            Dictionary containing:

            ``{"observs": np.array, "actions": np.array}``

        """
        data = {"observs": env_states.states, "actions": model_states.actions}
        return data

    def make_transitions(
        self, observs: numpy.ndarray, actions: numpy.ndarray
    ) -> Dict[str, numpy.ndarray]:
        """

        Sum the target action to the observations to obtain the new points, and \
        evaluate the reward and boundary conditions.

        Args:
            observs: Batch of points returned in the last step.
            actions: Perturbation that will be applied to ``observs``.

        Returns:
            Dictionary containing the information of the new points evaluated.

             ``{"states": new_points, "observs": new_points, "rewards": scalar array, \
             "oobs": boolean array}``

        """
        new_points = actions + observs
        oobs = self.calculate_oobs(points=new_points)
        rewards = self.function(new_points).flatten()
        data = {"states": new_points, "observs": new_points, "rewards": rewards, "oobs": oobs}
        return data

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the :class:`Function` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`EnvStates` instance describing the state of the :class:`Function`. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        oobs = numpy.zeros(batch_size, dtype=numpy.bool_)
        new_points = self.sample_bounds(batch_size=batch_size)
        rewards = self.function(new_points).flatten()
        new_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards,
            oobs=oobs,
            batch_size=batch_size,
        )
        return new_states

    def calculate_oobs(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        Determine if a given batch of vectors lie inside the function domain.

        Args:
            points: Array of batched vectors that will be checked to lie inside \
                    the :class:`Function` bounds.

        Returns:
            Array of booleans of length batch_size (points.shape[0]) that will \
            be ``True`` if a given point of the batch lies outside the bounds, \
            and ``False`` otherwise.

        """
        oobs = numpy.logical_not(self.bounds.points_in_bounds(points)).flatten()
        if self.custom_domain_check is not None:
            points_in_bounds = numpy.logical_not(oobs)
            oobs[points_in_bounds] = self.custom_domain_check(points[points_in_bounds])
        return oobs

    def sample_bounds(self, batch_size: int) -> numpy.ndarray:
        """
        Return a matrix of points sampled uniformly from the :class:`Function` \
        domain.

        Args:
            batch_size: Number of points that will be sampled.

        Returns:
            Array containing ``batch_size`` points that lie inside the \
            :class:`Function` domain, stacked across the first dimension.

        """
        new_points = numpy.zeros(tuple([batch_size]) + self.shape, dtype=numpy.float32)
        for i in range(batch_size):
            new_points[i, :] = self.random_state.uniform(
                low=self.bounds.low, high=self.bounds.high, size=self.shape
            )
        return new_points


class Minimizer:
    """Apply ``scipy.optimize.minimize`` to a :class:`Function`."""

    def __init__(self, function: Function, bounds=None, *args, **kwargs):
        """
        Initialize a :class:`Minimizer`.

        Args:
            function: :class:`Function` that will be minimized.
            bounds: :class:`Bounds` defining the domain of the minimization \
                    process. If it is ``None`` the :class:`Function` :class:`Bounds` \
                    will be used.
            *args: Passed to ``scipy.optimize.minimize``.
            **kwargs: Passed to ``scipy.optimize.minimize``.

        """
        self.env = function
        self.function = function.function
        self.bounds = self.env.bounds if bounds is None else bounds
        self.args = args
        self.kwargs = kwargs

    def minimize(self, x: numpy.ndarray):
        """
        Apply ``scipy.optimize.minimize`` to a single point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Optimization result object returned by ``scipy.optimize.minimize``.

        """

        def _optimize(_x):
            try:
                _x = _x.reshape((1,) + _x.shape)
                y = self.function(_x)
            except (ZeroDivisionError, RuntimeError):
                y = numpy.inf
            return y

        bounds = ScipyBounds(
            ub=self.bounds.high if self.bounds is not None else None,
            lb=self.bounds.low if self.bounds is not None else None,
        )
        return minimize(_optimize, x, bounds=bounds, *self.args, **self.kwargs)

    def minimize_point(self, x: numpy.ndarray) -> Tuple[numpy.ndarray, Scalar]:
        """
        Minimize the target function passing one starting point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Tuple containing a numpy array representing the best solution found, \
            and the numerical value of the function at that point.

        """
        optim_result = self.minimize(x)
        point = optim_result["x"]
        reward = float(optim_result["fun"])
        return point, reward

    def minimize_batch(self, x: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Minimize a batch of points.

        Args:
            x: Array representing a batch of points to be optimized, stacked \
               across the first dimension.

        Returns:
            Tuple of arrays containing the local optimum found for each point, \
            and an array with the values assigned to each of the points found.

        """
        result = numpy.zeros_like(x)
        rewards = numpy.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            new_x, reward = self.minimize_point(x[i, :])
            result[i, :] = new_x
            rewards[i, :] = float(reward)
        return result, rewards


class MinimizerWrapper(Function):
    """
    Wrapper that applies a local minimization process to the observations \
    returned by a :class:`Function`.
    """

    def __init__(self, function: Function, *args, **kwargs):
        """
        Initialize a :class:`MinimizerWrapper`.

        Args:
            function: :class:`Function` to be minimized after each step.
            *args: Passed to the internal :class:`Optimizer`.
            **kwargs: Passed to the internal :class:`Optimizer`.

        """
        self.env = function
        self.minimizer = Minimizer(function=self.env, *args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the wrapped environment."""
        return self.env.shape

    @property
    def function(self) -> Callable:
        """Return the function of the wrapped environment."""
        return self.env.function

    @property
    def bounds(self) -> Bounds:
        """Return the bounds of the wrapped environment."""
        return self.env.bounds

    @property
    def custom_domain_check(self) -> Callable:
        """Return the custom_domain_check of the wrapped environment."""
        return self.env.custom_domain_check

    def __getattr__(self, item):
        return self.env.__getattribute__(item)

    def __repr__(self):
        return self.env.__repr__()

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Perform a local optimization process to the observations returned after \
        calling ``step`` on the wrapped :class:`Function`.

        Args:
            model_states: :class:`StatesModel` corresponding to the :class:`Model` data.
            env_states: :class:`StatesEnv` containing the data where the function \
             will be evaluated.

        Returns:
            States containing the information that describes the new state of \
            the :class:`Function`.

        """
        env_states = super(MinimizerWrapper, self).step(
            model_states=model_states, env_states=env_states
        )
        new_points, rewards = self.minimizer.minimize_batch(env_states.observs)
        oobs = self.calculate_oobs(new_points)
        updated_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards.flatten(),
            oobs=oobs,
            batch_size=model_states.n,
        )
        return updated_states
