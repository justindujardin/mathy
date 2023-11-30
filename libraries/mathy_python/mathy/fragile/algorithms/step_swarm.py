import copy
from typing import Any, Callable, List, Tuple

import numpy

from fragile.core import Swarm, SwarmWrapper, Walkers
from fragile.core.base_classes import BaseModel, BaseTree
from fragile.core.states import OneWalker, StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import float_type, hash_numpy, running_in_ipython, Scalar, StateDict


class StepStatesWalkers(StatesWalkers):
    """
    :class:`StatesWalkers` that stores and clones information about the first \
    action selected during the search process, and its corresponding ``dt``.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a :walkers-st:`StepStatesWalkers`.

        Args:
            *args: Passed to :class:`StatesWalkers`.
            **kwargs: Passed to :class:`StatesWalkers`.

        """
        super(StepStatesWalkers, self).__init__(*args, **kwargs)
        self.init_actions = None
        self.init_dts = None

    def get_params_dict(self) -> StateDict:
        """
        Return the same StateDict as :class:`StatesWalkers` with two \
        additional attributes.

        **init_actions**: Used for storing the first action taken during \
                          a :swarm:`Swarm` run.
        **init_dts**: Used for storing the first action taken during \
                          a :swarm:`Swarm` run.

        """
        params_dict = super(StepStatesWalkers, self).get_params_dict()
        step_params = {"init_actions": {"dtype": float_type}, "init_dts": {"dtype": float_type}}
        params_dict.update(step_params)
        return params_dict

    def clone(self, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Perform clone like :class:`StatesWalkers` values and clone\
         ``init_action`` and ``init_dts``.
        """
        clone, compas = super(StepStatesWalkers, self).clone(**kwargs)
        self.init_actions[clone] = copy.deepcopy(self.init_actions[compas][clone])
        self.init_dts[clone] = copy.deepcopy(self.init_dts[compas][clone])
        return clone, compas

    def reset(self):
        """Reset the data of the :class:`StepStatesWalkers`."""
        super(StepStatesWalkers, self).reset()
        self.update(init_actions=numpy.zeros((len(self), 1)), init_dt=numpy.ones((len(self), 1)))


class StepWalkers(Walkers):
    """
    :walkers:`Walkers` that also keep track of the initial action \
    and initial dt sampled during a search process.

    This is done using a :class:`StepStatesWalkers` as its :walkers-st:`StatesWalkers`.
    """

    STATE_CLASS = StepStatesWalkers

    @property
    def states(self) -> StepStatesWalkers:
        """Return the :walkers-st:`StepStatesWalkers` where the walkers data is stored."""
        return self._states


class StoreInitAction(SwarmWrapper):
    """
    Wrapper that allows a swarm to keep track of the first action and first \
    dt sampled during a run.

    This is accomplished including an additional update of the walkers data that
    saves the initial actions and dts when running the first epoch of the run.
    """

    def __init__(self, swarm: Swarm):
        """
        Initialize a :swarm:`StoreInitAction`.

        Args:
            swarm: :class:`Swarm` that will be wrapped to keep track of the \
            storing of the first action and dt sampled.

        """
        super(StoreInitAction, self).__init__(swarm, name="_swarm")

    def step_walkers(self) -> None:
        """
        Make the walkers evolve to their next state sampling an action from the \
        :class:`Model` and applying it to the :class:`Environment`.
        """
        super(StoreInitAction, self).step_walkers()
        if self.epoch == 0:
            self.walkers.update_states(
                init_actions=copy.deepcopy(self.walkers.model_states.actions),
                init_dts=copy.deepcopy(self.walkers.get("dt", 1)),
            )

    def step_and_update_best(self) -> None:
        """
        Call the wrapped :class:`Swarm` ``step_and_update_best`` method, \
        but including the new ``step_walkers``.
        """
        return self._swarm.__class__.step_and_update_best(self)

    def run_step(self) -> None:
        """
        Call the wrapped :class:`Swarm` ``step_and_update_best`` method, \
        but including the new ``step_walkers``.
        """
        return self._swarm.__class__.run_step(self)

    def run(self, *args, **kwargs) -> None:
        """
        Call the wrapped :class:`Swarm` ``step_and_update_best`` method, \
        but including the new ``step_walkers``.
        """
        return self._swarm.__class__.run(self, *args, **kwargs)


class RootModel(BaseModel):
    """
    Model used to decide the discrete action that a root walker will take \
    after performing a search process with its internal :class:`Swarm`.
    """

    def get_params_dict(self) -> StateDict:
        """Return a :class:`StateDict` that defines discrete actions and time steps."""
        return {"actions": {"dtype": int}, "dt": {"dtype": int}}

    def predict(self, root_env_states: StatesEnv, walkers: StepWalkers,) -> StatesModel:
        """
        Sample the actions that the root walker will apply to the :env:`Environment`.

        Args:
            root_env_states: :env-st:`StatesEnv` class containing the data \
                            corresponding to the root walker of a :class:`StepSwarm`.
            walkers: :walkers:`StepWalkers` used by the internal swarm of a \
                     :class:`StepSwarm`.

        Returns:
            :class:`StatesModel` containing the ``actions`` and ``dt`` that the root walkers
            will use to step the :env:`Environment`.

        """
        raise NotImplementedError

    def reset(self, env_states: StatesEnv, walkers: StepWalkers):
        """Reset the internal data of the :class:`RootModel`."""
        pass


class MajorityDiscreteModel(RootModel):
    """
    :model:`Model` used to sample a discrete ``action`` and ``dt`` for a root walker.

    The predicted action will be the initial action taken by a majority of the \
    internal swarm's walkers. The returned dt will correspond to the smallest dt \
    of all the walkers that took the predicted action.
    """

    def predict(self, root_env_states: StatesEnv, walkers: StepWalkers,) -> StatesModel:
        """
        Select the most frequent ``init_action`` assigned to the internal swarm's walkers.

        The selected ``dt`` will be equal to the minimum ``init_dts`` among all \
        the walkers that sampled the selected ``init_action``.

        Args:
            root_env_states: :env-st:`StatesEnv` class containing the data \
                            corresponding to the root walker of a :class:`StepSwarm`.
            walkers: :walkers:`StepWalkers` used by the internal warm of a \
                     :class:`StepSwarm`.

        Returns:
            :class:`StatesModel` containing the ``actions`` and ``dt`` that the root walkers
            will use to step the :env:`Environment`.

        """
        init_actions = walkers.states.init_actions.flatten().astype(int)
        y = numpy.bincount(init_actions)
        most_used_action = numpy.nonzero(y)[0][0]
        root_model_states = StatesModel(
            batch_size=1, state_dict={"actions": {"dtype": int}, "dt": {"dtype": int}}
        )
        root_model_states.actions[:] = most_used_action
        if hasattr(root_model_states, "dt"):
            init_dts = walkers.states.init_dts.flatten().astype(int)
            index_dt = init_actions == most_used_action
            target_dt = init_dts[index_dt].min()
            root_model_states.dt[:] = target_dt
        return root_model_states


class FollowBestModel(RootModel):
    """
    :model:`Model` used to sample a discrete ``action`` and ``dt`` for a root walker.

    The selected action and dt will be equal to the ``init_actions`` and \
    ``init_dts`` assigned to the best walker found after the internal swarm run.
    """

    def predict(self, root_env_states: StatesEnv, walkers: StepWalkers,) -> StatesModel:
        """
        Select the ``init_action`` and ``init_dt`` of the best walker found \
        during the internal swarm run.

        Args:
            root_env_states: :env-st:`StatesEnv` class containing the data \
                            corresponding to the root walker of a :class:`StepSwarm`.
            walkers: :walkers:`StepWalkers` used by the internal swarm of a \
                     :class:`StepSwarm`.

        Returns:
            :class:`StatesModel` containing the ``actions`` and ``dt`` that the root walkers
            will use to step the :env:`Environment`.

        """
        init_actions = walkers.states.init_actions.flatten().astype(int)
        best_ix = walkers.get_best_index()
        root_model_states = StatesModel(
            batch_size=1, state_dict={"actions": {"dtype": int}, "dt": {"dtype": int}}
        )
        root_model_states.actions[:] = init_actions[best_ix]
        if hasattr(root_model_states, "dt"):
            target_dt = walkers.states.init_dt.flatten().astype(int)[best_ix]
            root_model_states.dt[:] = target_dt
        return root_model_states


class StepSwarm(Swarm):
    """
    The :class:`StepSwarm` is a :swarm:`Swarm` that builds a search tree to \
    sample one action.

    It implements the :class:`Swarm` methods, and can be used the same way a \
    :class:`Swarm` can be used.

    This search process for selecting an action if done using an **internal swarm**. \
    The internal swarm is a :class:`Swarm` that runs the a search process every \
    time that ``step_and_update_best`` is called using the **root walker** as \
    starting point. The internal swarm can be accessed as the ``internal_swarm`` \
    attribute of the :class:`StepSwarm`.

    The root walker represents the state of the search process of the :class:`StepSwarm`. \
    Even though a search process is run to select every action, the \
    :class:`StepSwarm` only follows one trajectory. The root walker tracks such \
    trajectory, and its actions can be sampled using a :class:`RootModel` that \
    is accessible in the ``root_model`` attribute.

    The root walker data can be accessed in the following attributes:

    - ``root_walker``: :class:`OneWalker` instance containing the data of the root walker.
    - ``root_env_states``: :class:`StatesEnv` containing the :env:`Environment` \
      data of the root walker.
    - ``root_model_states``: :class:`StatesModel` containing the data that will \
      be passed to the :class:`RootModel` to step the root walker.
    - ``root_walkers_states``: :class:`StatesWalkers` storing data of the root walker.

    """

    def __init__(
        self,
        n_walkers: int,
        step_epochs: int = None,
        root_model: Callable[[], RootModel] = MajorityDiscreteModel,
        tree: Callable[[], BaseTree] = None,
        report_interval: int = numpy.inf,
        show_pbar: bool = True,
        walkers: Callable[..., StepWalkers] = StepWalkers,
        swarm: Callable[..., Swarm] = Swarm,
        reward_limit: Scalar = None,
        max_epochs: int = None,
        accumulate_rewards: bool = True,
        minimize: bool = False,
        use_notebook_widget: bool = True,
        force_logging: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`StepSwarm`.

        This class can be initialized the same way as a :class:`Swarm`. All the \
        parameters except ``max_epochs`` and ``tree`` will be used to initialize \
        the internal swarm, and the :class:`StepSwarm` will use them when necessary.

        The internal swarm will be initialized with no tree, and with its \
        notebook widgets deactivated.

        Args:
            n_walkers: Number of walkers of the internal swarm.
            step_epochs: Number of epochs that the internal swarm will be run \
                         before sampling an action.
            root_model: Callable that returns a :class:`RootModel` that will be \
                        used to sample the actions and dt of the root walker.
            tree: Disabled for now. It will be used by the root walker.
            report_interval: Display the algorithm progress every \
                            ``report_interval`` epochs.
            show_pbar: If ``True`` A progress bar will display the progress of \
                       the algorithm run.
            walkers: A callable that returns an instance of :class:`StepWalkers`.
            swarm: A callable that returns an instance of :class:`Swarm` and \
                  takes as input all the corresponding parameters provided. It \
                  will be wrapped with a :class:`StoreInitAction` before \
                  assigning it to the ``internal_swarm`` attribute.
            reward_limit: The algorithm run will stop after reaching this \
                          reward value. If you are running a minimization process \
                          it will be considered the minimum reward possible, and \
                          if you are maximizing a reward it will be the maximum \
                          value.
            max_epochs: Maximum number of steps that the root walker is allowed \
                       to take.
            accumulate_rewards: If ``True`` the rewards obtained after transitioning \
                                to a new state will accumulate. If ``False`` only the last \
                                reward will be taken into account.
            minimize: If ``True`` the algorithm will perform a minimization \
                      process. If ``False`` it will be a maximization process.
            use_notebook_widget: If ``True`` and the class is running in an IPython \
                                kernel it will display the evolution of the swarm \
                                in a widget.
            force_logging: If ``True``, disable al ``ipython`` related behaviour.
            *args: Passed to ``swarm``.
            **kwargs: Passed to ``swarm``.

        """
        self.internal_swarm = StoreInitAction(
            swarm(
                max_epochs=step_epochs,
                show_pbar=False,
                report_interval=numpy.inf,
                n_walkers=n_walkers,
                tree=None,
                walkers=walkers,
                accumulate_rewards=accumulate_rewards,
                minimize=minimize,
                use_notebook_widget=False,
                *args,
                **kwargs
            )
        )
        self.internal_swarm.reset()
        self.root_model: RootModel = root_model()
        if reward_limit is None:
            reward_limit = -numpy.inf if self.internal_swarm.walkers.minimize else numpy.inf
        self.accumulate_rewards = accumulate_rewards
        self._max_epochs = int(max_epochs)
        self.reward_limit = reward_limit
        self.show_pbar = show_pbar
        self.report_interval = report_interval
        self.tree = tree() if tree is not None else tree
        self._epoch = 0
        self._walkers: StepWalkers = self.internal_swarm.walkers
        self._model = self.internal_swarm.model
        self._env = self.internal_swarm.env
        self.cum_reward = numpy.NINF
        self.minimize = minimize
        self.root_model_states = self.walkers.model_states[0]
        self.root_env_states = self.walkers.env_states[0]
        self.root_walkers_states = self.walkers.states[0]
        self.root_walker = OneWalker(
            reward=self.root_env_states.rewards[0],
            observ=self.root_env_states.observs[0],
            state=self.root_env_states.states[0],
        )
        self._notebook_container = None
        self._use_notebook_widget = use_notebook_widget
        self._ipython_mode = running_in_ipython() and not force_logging
        self.setup_notebook_container()

    def __repr__(self):
        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            init_actions = self.internal_swarm.walkers.states.init_actions.flatten()
            y = numpy.bincount(init_actions.astype(int))
            ii = numpy.nonzero(y)[0]
            string = str(self.root_walker)
            string += "\n Init actions [action, count]: \n%s" % numpy.vstack((ii, y[ii])).T
            return string

    @property
    def max_epochs(self) -> int:
        """Return the maximum number of epochs allowed."""
        return self._max_epochs

    @property
    def best_state(self) -> numpy.ndarray:
        """Return the state of the best walker found in the current algorithm run."""
        return self.root_walker.states[0]

    @property
    def best_reward(self) -> Scalar:
        """Return the reward of the best walker found in the current algorithm run."""
        return self.root_walker.rewards[0]

    @property
    def best_id(self) -> int:
        """
        Return the id (hash value of the state) of the best walker found in the \
        current algorithm run.
        """
        return self.root_walker.id_walkers[0]

    @property
    def best_obs(self) -> numpy.ndarray:
        """
        Return the observation corresponding to the best walker found in the \
        current algorithm run.
        """
        return self.root_walker.observs[0]

    def get(self, name, default: Any = None) -> Any:
        """
        Access attributes of the swarm.

        Epoch and best attributes will be read from the root walker. The other \
        attributes will be read from the internal swarm.
        """
        if "best" in name or "epoch" in name:
            return getattr(self, name)
        return self.internal_swarm.get(name, default)

    def reset(
        self,
        root_walker: OneWalker = None,
        walkers_states: StatesWalkers = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
    ):
        """
        Reset the :class:`fragile.Walkers`, the :class:`Environment`, the \
        :class:`Model` and clear the internal data to start a new search process.

        Args:
            root_walker: Walker representing the initial state of the search. \
                         The walkers will be reset to this walker, and it will \
                         be added to the root of the :class:`StateTree` if any.
            model_states: :class:`StatesModel` that define the initial state of \
                          the :class:`Model`.
            env_states: :class:`StatesEnv` that define the initial state of \
                        the :class:`Environment`.
            walkers_states: :class:`StatesWalkers` that define the internal \
                            states of the :class:`Walkers`.

        """
        self._epoch = 0
        self.internal_swarm.reset(
            root_walker=root_walker,
            walkers_states=walkers_states,
            env_states=env_states,
            model_states=model_states,
        )
        # Reset root data
        best_index = self.walkers.get_best_index()
        self.root_model_states = self.walkers.model_states[best_index]
        self.root_env_states = self.walkers.env_states[best_index]
        self.root_walkers_states = self.walkers.states[best_index]
        self.root_walker = OneWalker(
            reward=self.root_env_states.rewards[best_index],
            observ=self.root_env_states.observs[best_index],
            state=self.root_env_states.states[best_index],
        )
        if self.tree is not None:
            self.tree.reset(
                root_id=self.best_id,
                env_states=self.root_env_states[0],
                model_states=self.root_model_states[0],
                walkers_states=self.root_walkers_states[0],
            )

    def calculate_end_condition(self) -> bool:
        """Implement the logic for deciding if the algorithm has finished. \
        The algorithm will stop if it returns True."""
        max_reward_reached = (
            self.best_reward < self.reward_limit
            if self.minimize
            else self.best_reward > self.reward_limit
        )
        max_epochs_reached = self.epoch > self.max_epochs
        best_is_oob = self.root_env_states.oobs[0]
        return best_is_oob or max_epochs_reached or max_reward_reached

    def update_tree(self, states_ids: List[int]) -> None:
        """
        Add a list of walker states represented by `states_ids` to the :class:`Tree`.

        Args:
            states_ids: list containing the ids of the new states added.
        """
        if self.tree is not None:
            self.tree.add_states(
                parent_ids=states_ids,
                env_states=self.root_env_states,
                model_states=self.root_model_states,
                walkers_states=self.root_walkers_states,
                n_iter=int(self.epoch),
            )

    def balance_and_prune(self) -> None:
        """Do nothing. Only the internal swarm balances its walkers."""
        pass

    def step_and_update_best(self) -> None:
        """Run the internal swarm."""
        self.internal_swarm.run(root_walker=self.root_walker)

    def run_step(self) -> None:
        """
        Compute one iteration of the :class:`Swarm` evolution process and \
        update all the data structures.
        """
        self.step_and_update_best()
        self.step_root_state()

    def step_root_state(self):
        """Make the state transition of the root state."""
        model_states = self.root_model.predict(
            root_env_states=self.root_env_states, walkers=self.walkers
        )
        parent_id = copy.copy(self.best_id)
        new_env_states = self.env.step(model_states=model_states, env_states=self.root_env_states)
        self.update_states(new_env_states, model_states)
        self.update_tree(states_ids=[parent_id])

    def update_states(self, env_states, model_states):
        """Update the data of the root state."""
        self.root_env_states.update(other=env_states)
        self.root_model_states.update(other=model_states)
        if self.accumulate_rewards:
            cum_rewards = self.root_walkers_states.cum_rewards
            cum_rewards = cum_rewards + self.root_env_states.rewards
        else:
            cum_rewards = self.root_env_states.rewards
        self.root_walkers_states.update(
            cum_rewards=cum_rewards,
            id_walkers=numpy.array([hash_numpy(self.root_env_states.states[0])]),
        )

        self.root_walker = OneWalker(
            reward=copy.deepcopy(cum_rewards[0]),
            observ=copy.deepcopy(self.root_env_states.observs[0]),
            state=copy.deepcopy(self.root_env_states.states[0]),
        )


class StepToBest(StepSwarm):
    """
    The :swarm:`StatesToBest` uses an internal :class:`Swarm` to \
    perform a search process and steps the root walker to the best state found \
    by the internal swarm.
    """

    def run_step(self) -> None:
        """
        Step the internal Swarm and update the root walker to match the state \
        of the best walker found.
        """
        self.step_and_update_best()
        self.step_root_state()

    def step_root_state(self):
        """Step the root state to the best walker found during the run of the internal Swarm."""
        ib = self.walkers.states.in_bounds
        in_bounds_rewards = self.walkers.get("cum_rewards")[ib]
        if len(in_bounds_rewards) == 0:
            best_ix = 0
        else:
            best_ix_filtered = int(
                in_bounds_rewards.argmin() if self.walkers.minimize else in_bounds_rewards.argmax()
            )
            best_ix = int(numpy.arange(self.walkers.n)[ib][best_ix_filtered])
        self.root_env_states = self.walkers.env_states[best_ix]
        self.root_walkers_states = self.walkers.states[best_ix]
        self.update_states()
        self.update_tree([self.best_id])

    def update_states(self):
        """Update the data of the root walker after an internal Swarm iteration has finished."""
        # The accumulation of rewards is already done in the internal Swarm
        cum_rewards = self.root_walkers_states.cum_rewards
        self.root_walkers_states.update(
            cum_rewards=cum_rewards,
            id_walkers=numpy.array([hash_numpy(self.root_env_states.states[0])]),
        )
        self.root_walker = OneWalker(
            reward=copy.deepcopy(cum_rewards[0]),
            observ=copy.deepcopy(self.root_env_states.observs[0]),
            state=copy.deepcopy(self.root_env_states.states[0]),
        )
