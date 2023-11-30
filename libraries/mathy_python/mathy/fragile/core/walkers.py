import copy
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy

from fragile.core.base_classes import BaseCritic, BaseWalkers
from fragile.core.functions import relativize
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import (
    DistanceFunction,
    float_type,
    hash_numpy,
    Scalar,
    StateDict,
    statistics_from_array,
)


class SimpleWalkers(BaseWalkers):
    """
    This class is in charge of performing all the mathematical operations involved in evolving a \
    cloud of walkers.

    """

    STATE_CLASS = StatesWalkers

    def __init__(
        self,
        n_walkers: int,
        env_state_params: StateDict,
        model_state_params: StateDict,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        accumulate_rewards: bool = True,
        max_epochs: int = None,
        distance_function: Optional[
            Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
        ] = None,
        ignore_clone: Optional[Dict[str, Set[str]]] = None,
        **kwargs
    ):
        """
        Initialize a new `Walkers` instance.

        Args:
            n_walkers: Number of walkers of the instance.
            env_state_params: Dictionary to instantiate the States of an :class:`Environment`.
            model_state_params: Dictionary to instantiate the States of a :class:`Model`.
            reward_scale: Regulates the importance of the reward. Recommended to \
                          keep in the [0, 5] range. Higher values correspond to \
                          higher importance.
            distance_scale: Regulates the importance of the distance. Recommended to \
                            keep in the [0, 5] range. Higher values correspond to \
                            higher importance.
            accumulate_rewards: If ``True`` the rewards obtained after transitioning \
                                to a new state will accumulate. If ``False`` only the last \
                                reward will be taken into account.
            distance_function: Function to compute the distances between two \
                               groups of walkers. It will be applied row-wise \
                               to the walkers observations and it will return a \
                               vector of scalars. Defaults to l2 norm.
            ignore_clone: Dictionary containing the attribute values that will \
                          not be cloned. Its keys can be be either "env", of \
                          "model", to reference the `env_states` and the \
                          `model_states`. Its values are a set of strings with \
                          the names of the attributes that will not be cloned.
            max_epochs: Maximum number of iterations that the walkers are allowed \
                       to perform.
            kwargs: Additional attributes stored in the :class:`StatesWalkers`.

        """
        super(SimpleWalkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
            accumulate_rewards=accumulate_rewards,
            max_epochs=max_epochs,
        )

        def l2_norm(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
            return numpy.sqrt(numpy.sum((x - y) ** 2, axis=1))

        self._model_states = StatesModel(state_dict=model_state_params, batch_size=n_walkers)
        self._env_states = StatesEnv(state_dict=env_state_params, batch_size=n_walkers)
        self._states = self.STATE_CLASS(batch_size=n_walkers, **kwargs)
        self.distance_function = distance_function if distance_function is not None else l2_norm
        self.reward_scale = reward_scale
        self.distance_scale = distance_scale
        self._id_counter = 0
        self.ignore_clone = ignore_clone if ignore_clone is not None else {}

    def __repr__(self) -> str:
        """Print all the data involved in the current run of the algorithm."""
        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            try:
                text = self._print_stats()
                text += "Walkers States: {}\n".format(self._repr_state(self._states))
                text += "Environment States: {}\n".format(self._repr_state(self._env_states))
                text += "Model States: {}\n".format(self._repr_state(self._model_states))
                return text
            except Exception:
                return super(SimpleWalkers, self).__repr__()

    def _print_stats(self) -> str:
        """Print several statistics of the current state of the swarm."""
        text = "{} iteration {} Out of bounds walkers: {:.2f}% Cloned: {:.2f}%\n\n".format(
            self.__class__.__name__,
            self.epoch,
            100 * self.env_states.oobs.sum() / self.n,
            100 * self.states.will_clone.sum() / self.n,
        )
        return text

    def get(self, name: str, default: Any = None) -> Any:
        """Access attributes of the :class:`Swarm` and its children."""
        if hasattr(self.states, name):
            return getattr(self.states, name)
        elif hasattr(self.env_states, name):
            return getattr(self.env_states, name)
        elif hasattr(self.model_states, name):
            return getattr(self.model_states, name)
        elif hasattr(self, name):
            return getattr(self, name)
        return default

    def ids(self) -> numpy.ndarray:
        """
        Return a list of unique ids for each walker state.

        The returned ids are integers representing the hash of the different states.
        """
        return numpy.array(self.env_states.hash_values("states"))

    def update_ids(self):
        """Update the unique id of each walker and store it in the :class:`StatesWalkers`."""
        self.states.update(id_walkers=self.ids().copy())

    @property
    def states(self) -> StatesWalkers:
        """Return the `StatesWalkers` class that contains the data used by the instance."""
        return self._states

    @property
    def env_states(self) -> StatesEnv:
        """Return the `States` class that contains the data used by the :class:`Environment`."""
        return self._env_states

    @property
    def model_states(self) -> StatesModel:
        """Return the `States` class that contains the data used by a Model."""
        return self._model_states

    @property
    def best_state(self) -> numpy.ndarray:
        """Return the state of the best walker found in the current algorithm run."""
        return self.states.best_state

    @property
    def best_reward(self) -> Scalar:
        """Return the reward of the best walker found in the current algorithm run."""
        return self.states.best_reward

    @property
    def best_id(self) -> int:
        """
        Return the id (hash value of the state) of the best walker found in the \
        current algorithm run.
        """
        return self.states.best_id

    @property
    def best_obs(self) -> numpy.ndarray:
        """
        Return the observation corresponding to the best walker found in the \
        current algorithm run.
        """
        return self.states.best_obs

    def calculate_end_condition(self) -> bool:
        """
        Process data from the current state to decide if the iteration process should stop.

        Returns:
            Boolean indicating if the iteration process should be finished. ``True`` means \
            it should be stopped, and ``False`` means it should continue.

        """
        non_terminal_states = numpy.logical_not(self.env_states.terminals)
        all_non_terminal_out_of_bounds = self.env_states.oobs[non_terminal_states].all()
        max_epochs_reached = self.epoch >= self.max_epochs
        all_in_bounds_are_terminal = self.env_states.terminals[self.states.in_bounds].all()
        return max_epochs_reached or all_non_terminal_out_of_bounds or all_in_bounds_are_terminal

    def calculate_distances(self) -> None:
        """Calculate the corresponding distance function for each observation with \
        respect to another observation chosen at random.

        The internal :class:`StateWalkers` is updated with the relativized distance values.
        """
        # TODO(guillemdb): Check if self.get_in_bounds_compas() works better.
        compas_ix = numpy.random.permutation(numpy.arange(self.n))
        obs = self.env_states.observs.reshape(self.n, -1)
        distances = self.distance_function(obs, obs[compas_ix])
        distances = relativize(distances.flatten())
        self.update_states(distances=distances, compas_dist=compas_ix)

    def calculate_virtual_reward(self) -> None:
        """
        Calculate the virtual reward and update the internal state.

        The cumulative_reward is transformed with the relativize function. \
        The distances stored in the :class:`StatesWalkers` are already transformed.
        """
        processed_rewards = relativize(self.states.cum_rewards)
        virt_rw = (
            processed_rewards ** self.reward_scale * self.states.distances ** self.distance_scale
        )
        self.update_states(virtual_rewards=virt_rw, processed_rewards=processed_rewards)

    def get_in_bounds_compas(self) -> numpy.ndarray:
        """
        Return the indexes of walkers inside bounds chosen at random.

        Returns:
            Numpy array containing the int indexes of in bounds walkers chosen at \
            random with replacement. Its length is equal to the number of walkers.

        """
        if not self.states.in_bounds.any():  # No need to sample if all walkers are dead.
            return numpy.arange(self.n)
        alive_indexes = numpy.arange(self.n, dtype=int)[self.states.in_bounds]
        compas_ix = self.random_state.permutation(alive_indexes)
        compas = self.random_state.choice(compas_ix, self.n, replace=True)
        compas[: len(compas_ix)] = compas_ix
        return compas

    def update_clone_probs(self) -> None:
        """
        Calculate the new probability of cloning for each walker.

        Updates the :class:`StatesWalkers` with both the probability of cloning \
        and the index of the randomly chosen companions that were selected to \
        compare the virtual rewards.
        """
        all_virtual_rewards_are_equal = (
            self.states.virtual_rewards == self.states.virtual_rewards[0]
        ).all()
        if all_virtual_rewards_are_equal:
            clone_probs = numpy.zeros(self.n, dtype=float_type)
            compas_ix = numpy.arange(self.n)
        else:
            compas_ix = self.get_in_bounds_compas()
            companions = self.states.virtual_rewards[compas_ix]
            # This value can be negative!!
            clone_probs = (companions - self.states.virtual_rewards) / self.states.virtual_rewards
        self.update_states(clone_probs=clone_probs, compas_clone=compas_ix)

    def balance(self) -> Tuple[set, set]:
        """
        Perform an iteration of the FractalAI algorithm for balancing the \
        walkers distribution.

        It performs the necessary calculations to determine which walkers will clone, \
        and performs the cloning process.

        Returns:
            A tuple containing two sets: The first one represent the unique ids \
            of the states for each walker at the start of the iteration. The second \
            one contains the ids of the states after the cloning process.

        """
        old_ids = set(self.states.id_walkers.copy())
        self.states.in_bounds = numpy.logical_not(self.env_states.oobs)
        self.calculate_distances()
        self.calculate_virtual_reward()
        self.update_clone_probs()
        self.clone_walkers()
        new_ids = set(self.states.id_walkers.copy())
        return old_ids, new_ids

    def clone_walkers(self) -> None:
        """
        Sample the clone probability distribution and clone the walkers accordingly.

        This function will update the internal :class:`StatesWalkers`, \
        :class:`StatesEnv`, and :class:`StatesModel`.
        """
        will_clone = self.states.clone_probs > self.random_state.random_sample(self.n)
        will_clone[self.env_states.oobs] = True  # Out of bounds walkers always clone
        self.update_states(will_clone=will_clone)
        clone, compas = self.states.clone()
        self._env_states.clone(
            will_clone=clone, compas_ix=compas, ignore=self.ignore_clone.get("env")
        )
        self._model_states.clone(
            will_clone=clone, compas_ix=compas, ignore=self.ignore_clone.get("model")
        )

    def reset(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
    ) -> None:
        """
        Restart all the internal states involved in the algorithm iteration.

        After reset a new run of the algorithm will be ready to be launched.
        """
        if walkers_states is not None:
            self.states.update(walkers_states)
        else:
            self.states.reset()
        self.update_states(env_states=env_states, model_states=model_states)
        self._epoch = 0

    def update_states(
        self, env_states: StatesEnv = None, model_states: StatesModel = None, **kwargs
    ):
        """
        Update the States variables that do not contain internal data and \
        accumulate the rewards in the internal states if applicable.

        Args:
            env_states: States containing the data associated with the Environment.
            model_states: States containing data associated with the Environment.
            **kwargs: Internal states will be updated via keyword arguments.

        """
        if kwargs:
            if kwargs.get("rewards") is not None:
                self._accumulate_and_update_rewards(kwargs["rewards"])
                del kwargs["rewards"]
            self.states.update(**kwargs)
        if isinstance(env_states, StatesEnv):
            self._env_states.update(env_states)
            if hasattr(env_states, "rewards"):
                self._accumulate_and_update_rewards(env_states.rewards)
        if isinstance(model_states, StatesModel):
            self._model_states.update(model_states)
        self.update_ids()

    def _accumulate_and_update_rewards(self, rewards: numpy.ndarray):
        """
        Use as reward either the sum of all the rewards received during the \
        current run, or use the last reward value received as reward.

        Args:
            rewards: Array containing the last rewards received by every walker.
        """
        if self._accumulate_rewards:
            if not isinstance(self.states.get("cum_rewards"), numpy.ndarray):
                cum_rewards = numpy.zeros(self.n)
            else:
                cum_rewards = self.states.cum_rewards
            cum_rewards = cum_rewards + rewards
        else:
            cum_rewards = rewards
        self.update_states(cum_rewards=cum_rewards)

    @staticmethod
    def _repr_state(state):
        string = "\n"
        for k, v in state.items():
            if k in ["observs", "states", "id_walkers", "best_id"]:
                continue
            shape = v.shape if hasattr(v, "shape") else None
            new_str = (
                "{}: shape {} Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(
                    k, shape, *statistics_from_array(v)
                )
                if isinstance(v, numpy.ndarray) and "best" not in k
                else ("%s %s\n" % (k, v if not isinstance(v, numpy.ndarray) else v.flatten()))
            )
            string += new_str
        return string

    def fix_best(self):
        """Ensure the best state found is assigned to the last walker of the \
        swarm, so walkers can always choose to clone to the best state."""
        pass


class Walkers(SimpleWalkers):
    """
    The Walkers is a data structure that takes care of all the data involved \
    in making a Swarm evolve.
    """

    def __init__(
        self,
        n_walkers: int,
        env_state_params: StateDict,
        model_state_params: StateDict,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        max_epochs: int = None,
        accumulate_rewards: bool = True,
        distance_function: Optional[DistanceFunction] = None,
        ignore_clone: Optional[Dict[str, Set[str]]] = None,
        critic: Optional[BaseCritic] = None,
        minimize: bool = False,
        best_walker: Optional[Tuple[numpy.ndarray, numpy.ndarray, Scalar]] = None,
        reward_limit: float = None,
        **kwargs
    ):
        """
        Initialize a :class:`Walkers`.

        Args:
            n_walkers: Number of walkers of the instance.
            env_state_params: Dictionary to instantiate the States of an :class:`Environment`.
            model_state_params: Dictionary to instantiate the States of a :class:`Model`.
            reward_scale: Regulates the importance of the reward. Recommended to \
                          keep in the [0, 5] range. Higher values correspond to \
                          higher importance.
            distance_scale: Regulates the importance of the distance. Recommended to \
                            keep in the [0, 5] range. Higher values correspond to \
                            higher importance.
            max_epochs: Maximum number of iterations that the walkers are allowed \
                       to perform.
            accumulate_rewards: If ``True`` the rewards obtained after transitioning \
                                to a new state will accumulate. If ``False`` only the last \
                                reward will be taken into account.
            distance_function: Function to compute the distances between two \
                               groups of walkers. It will be applied row-wise \
                               to the walkers observations and it will return a \
                               vector of scalars. Defaults to l2 norm.
            ignore_clone: Dictionary containing the attribute values that will \
                          not be cloned. Its keys can be be either "env", of \
                          "model", to reference the `env_states` and the \
                          `model_states`. Its values are a set of strings with \
                          the names of the attributes that will not be cloned.
            critic: :class:`Critic` that will be used to calculate custom rewards.
            minimize: If ``True`` the algorithm will perform a minimization \
                      process. If ``False`` it will be a maximization process.
            best_walker: Tuple containing the best state and reward that will \
                        be used as the initial best values found.
            reward_limit: The algorithm run will stop after reaching this \
                          reward value. If you are running a minimization process \
                          it will be considered the minimum reward possible, and \
                          if you are maximizing a reward it will be the maximum \
                          value.
            kwargs: Additional attributes stored in the :class:`StatesWalkers`.

        """
        # Add data specific to the child class in the StatesWalkers class as new attributes.
        if critic is not None:
            kwargs["critic_score"] = kwargs.get("critic_score", numpy.zeros(n_walkers))
        self.dtype = float_type
        if best_walker is not None:
            best_state, best_obs, best_reward = best_walker
            best_id = hash_numpy(best_state)
        else:
            best_state, best_obs, best_reward, best_id = (None, None, -numpy.inf, None)
        super(Walkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
            reward_scale=reward_scale,
            distance_scale=distance_scale,
            max_epochs=max_epochs,
            accumulate_rewards=accumulate_rewards,
            distance_function=distance_function,
            ignore_clone=ignore_clone,
            best_reward=best_reward,
            best_obs=best_obs,
            best_state=best_state,
            best_id=best_id,
            **kwargs
        )
        self.critic = critic
        self.minimize = minimize
        self.efficiency = 0
        self._min_entropy = 0
        if reward_limit is None:
            reward_limit = -numpy.inf if self.minimize else numpy.inf
        self.reward_limit = reward_limit

    def __repr__(self):
        text = "\nBest reward found: {:.4f} , efficiency {:.3f}, Critic: {}\n".format(
            float(self.states.best_reward), self.efficiency, self.critic
        )
        return text + super(Walkers, self).__repr__()

    def calculate_end_condition(self) -> bool:
        """
        Process data from the current state to decide if the iteration process should stop.

        Returns:
            Boolean indicating if the iteration process should be finished. ``True`` means \
            it should be stopped, and ``False`` means it should continue.

        """
        end_condition = super(Walkers, self).calculate_end_condition()
        reward_limit_reached = (
            self.states.best_reward < self.reward_limit
            if self.minimize
            else self.states.best_reward > self.reward_limit
        )
        return end_condition or reward_limit_reached

    def calculate_virtual_reward(self):
        """Apply the virtual reward formula to account for all the different goal scores."""
        rewards = -1 * self.states.cum_rewards if self.minimize else self.states.cum_rewards
        processed_rewards = relativize(rewards)
        score_reward = processed_rewards ** self.reward_scale
        score_dist = self.states.distances ** self.distance_scale
        virt_rw = score_reward * score_dist
        dist_prob = score_dist / score_dist.sum()
        reward_prob = score_reward / score_reward.sum()
        total_entropy = numpy.prod(2 - dist_prob ** reward_prob)
        self._min_entropy = numpy.prod(2 - reward_prob ** reward_prob)
        self.efficiency = self._min_entropy / total_entropy
        self.update_states(virtual_rewards=virt_rw, processed_rewards=processed_rewards)
        if self.critic is not None:
            critic_states = self.critic.calculate(
                walkers_states=self.states,
                model_states=self.model_states,
                env_states=self.env_states,
            )
            self.states.update(other=critic_states)
            virt_rew = self.states.virtual_rewards * self.states.critic
        else:
            virt_rew = self.states.virtual_rewards
        self.states.update(virtual_rewards=virt_rew)

    def balance(self):
        """Perform FAI iteration to clone the states."""
        self.update_best()
        returned = super(Walkers, self).balance()
        if self.critic is not None:
            critic_states = self.critic.update(
                walkers_states=self.states,
                model_states=self.model_states,
                env_states=self.env_states,
            )
            self.states.update(other=critic_states)
        return returned

    def get_best_index(self) -> int:
        """
        Return the index of the best state present in the :class:`Walkers` \
        that is considered alive (inside the boundary conditions of the problem). \
        If no walker is alive it will return the index of the last walker, which \
        corresponds with the best state found.
        """
        rewards = self.states.cum_rewards[self.states.in_bounds]
        if len(rewards) == 0:
            return self.n - 1
        best = rewards.min() if self.minimize else rewards.max()
        idx = (self.states.cum_rewards == best).astype(int)
        ix = idx.argmax()
        return int(ix)

    def update_best(self):
        """Keep track of the best state found and its reward."""
        ix = self.get_best_index()
        best_obs = self.env_states.observs[ix].copy()
        best_reward = float(self.states.cum_rewards[ix])
        best_state = self.env_states.states[ix].copy()
        best_is_in_bounds = not bool(self.env_states.oobs[ix])
        has_improved = (
            self.states.best_reward > best_reward
            if self.minimize
            else self.states.best_reward < best_reward
        )
        if has_improved and best_is_in_bounds:
            self.states.update(
                best_reward=best_reward,
                best_state=best_state,
                best_obs=best_obs,
                best_id=copy.copy(self.states.id_walkers[ix]),
            )

    def fix_best(self):
        """Ensure the best state found is assigned to the last walker of the \
        swarm, so walkers can always choose to clone to the best state."""
        if self.states.best_reward is not None:
            self.env_states.observs[-1] = self.states.best_obs.copy()
            self.states.cum_rewards[-1] = float(self.states.best_reward)
            self.states.id_walkers[-1] = copy.copy(self.states.best_id)
            self.env_states.states[-1] = self.states.best_state.copy()

    def reset(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
    ):
        """
        Reset a :class:`Walkers` and clear the internal data to start a \
        new search process.

        Restart all the variables needed to perform the fractal evolution process.

        Args:
            model_states: :class:`StatesModel` that define the initial state of the environment.
            env_states: :class:`StatesEnv` that define the initial state of the model.
            walkers_states: :class:`StatesWalkers` that define the internal states of the walkers.

        """
        super(Walkers, self).reset(
            env_states=env_states, model_states=model_states, walkers_states=walkers_states
        )
        best_ix = (
            self.env_states.rewards.argmin() if self.minimize else self.env_states.rewards.argmax()
        )
        self.states.update(
            best_reward=copy.deepcopy(self.env_states.rewards[best_ix]),
            best_obs=copy.deepcopy(self.env_states.observs[best_ix]),
            best_state=copy.deepcopy(self.env_states.states[best_ix]),
            best_id=copy.deepcopy(self.states.id_walkers[best_ix]),
        )
        if self.critic is not None:
            critic_score = self.critic.reset(
                env_states=self.env_states, model_states=model_states, walker_states=walkers_states
            )
            self.states.update(critic_score=critic_score)
