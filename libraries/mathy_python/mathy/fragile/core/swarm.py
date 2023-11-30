import copy
import logging
from typing import Any, Callable, Iterable, List

import numpy

from fragile.core.base_classes import (
    BaseCritic,
    BaseEnvironment,
    BaseModel,
    BaseSwarm,
)
from fragile.core.states import OneWalker, StatesEnv, StatesModel, StatesWalkers
from fragile.core.tree import HistoryTree
from fragile.core.utils import running_in_ipython, Scalar
from fragile.core.walkers import Walkers


class Swarm(BaseSwarm):
    """
    The Swarm is in charge of performing a fractal evolution process.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to run the Swarm evolution algorithm.
    """

    _log = logging.getLogger("Swarm")

    def __init__(
        self,
        n_walkers: int,
        env: Callable[[], BaseEnvironment],
        model: Callable[[BaseEnvironment], BaseModel],
        walkers: Callable[..., Walkers] = Walkers,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        tree: Callable[[], HistoryTree] = None,
        report_interval: int = numpy.inf,
        show_pbar: bool = True,
        use_notebook_widget: bool = True,
        force_logging: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`Swarm`.

        Args:
            n_walkers: Number of walkers of the swarm.
            env: A callable that returns an instance of an Environment.
            model: A callable that returns an instance of a Model.
            walkers: A callable that returns an instance of BaseWalkers.
            reward_scale: Virtual reward exponent for the reward score.
            distance_scale: Virtual reward exponent for the distance score.
            tree: class:`StatesTree` that keeps track of the visited states.
            report_interval: Display the algorithm progress every ``report_interval`` epochs.
            show_pbar: If ``True`` A progress bar will display the progress of \
                       the algorithm run.
            use_notebook_widget: If ``True`` and the class is running in an IPython \
                                kernel it will display the evolution of the swarm \
                                in a widget.
            force_logging: If ``True``, disable al ``ipython`` related behaviour.
            *args: Additional args passed to init_swarm.
            **kwargs: Additional kwargs passed to init_swarm.

        """
        self._prune_tree = False
        self._epoch = 0
        self.show_pbar = show_pbar
        self.report_interval = report_interval
        super(Swarm, self).__init__(
            walkers=walkers,
            env=env,
            model=model,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            distance_scale=distance_scale,
            tree=tree,
            *args,
            **kwargs
        )
        self._notebook_container = None
        self._use_notebook_widget = use_notebook_widget
        self._ipython_mode = running_in_ipython() and not force_logging
        self.setup_notebook_container()

    def __len__(self) -> int:
        return self.walkers.n

    def __repr__(self) -> str:
        walkers_data = self.walkers.__repr__()
        tree_data = self.tree.__repr__() if self.tree is not None else ""
        return walkers_data + tree_data

    @property
    def env(self) -> BaseEnvironment:
        """All the simulation code (problem specific) will be handled here."""
        return self._env

    @property
    def model(self) -> BaseModel:
        """
        All the policy and random perturbation code (problem specific) will \
        be handled here.
        """
        return self._model

    @property
    def walkers(self) -> Walkers:
        """
        Access the :class:`Walkers` in charge of implementing the FAI \
        evolution process.
        """
        return self._walkers

    @property
    def best_state(self) -> numpy.ndarray:
        """Return the state of the best walker found in the current algorithm run."""
        return self.walkers.best_state

    @property
    def best_reward(self) -> Scalar:
        """Return the reward of the best walker found in the current algorithm run."""
        return self.walkers.best_reward

    @property
    def best_id(self) -> int:
        """
        Return the id (hash value of the state) of the best walker found in the \
        current algorithm run.
        """
        return self.walkers.best_id

    @property
    def best_obs(self) -> numpy.ndarray:
        """
        Return the observation corresponding to the best walker found in the \
        current algorithm run.
        """
        return self.walkers.best_obs

    @property
    def critic(self) -> BaseCritic:
        """Return the :class:`Critic` of the walkers."""
        return self._walkers.critic

    def get(self, name: str, default: Any = None) -> Any:
        """Access attributes of the :class:`Swarm` and its children."""
        if hasattr(self.walkers.states, name):
            return getattr(self.walkers.states, name)
        elif hasattr(self.walkers.env_states, name):
            return getattr(self.walkers.env_states, name)
        elif hasattr(self.walkers.model_states, name):
            return getattr(self.walkers.model_states, name)
        elif hasattr(self.walkers, name):
            return getattr(self.walkers, name)
        elif hasattr(self, name):
            return getattr(self, name)
        return default

    def init_swarm(
        self,
        env_callable: Callable[[], BaseEnvironment],
        model_callable: Callable[[BaseEnvironment], BaseModel],
        walkers_callable: Callable[..., Walkers],
        n_walkers: int,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        tree: Callable[[], HistoryTree] = None,
        prune_tree: bool = True,
        *args,
        **kwargs
    ):
        """
        Initialize and set up all the necessary internal variables to run the swarm.

        This process involves instantiating the Swarm, the Environment and the \
        model.

        Args:
            env_callable: A callable that returns an instance of an
                :class:`fragile.Environment`.
            model_callable: A callable that returns an instance of a
                :class:`fragile.Model`.
            walkers_callable: A callable that returns an instance of
                :class:`fragile.Walkers`.
            n_walkers: Number of walkers of the swarm.
            reward_scale: Virtual reward exponent for the reward score.
            distance_scale: Virtual reward exponent for the distance score.
            tree: class:`StatesTree` that keeps track of the visited states.
            prune_tree: If `tree` is `None` it has no effect. If true, \
                       store in the :class:`Tree` only the past history of alive \
                        walkers, and discard the branches with leaves that have \
                        no walkers.
            args: Passed to ``walkers_callable``.
            kwargs: Passed to ``walkers_callable``.

        Returns:
            None.

        """
        self._env: BaseEnvironment = env_callable()
        self._model: BaseModel = model_callable(self._env)

        model_params = self._model.get_params_dict()
        env_params = self._env.get_params_dict()
        self._walkers: Walkers = walkers_callable(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            distance_scale=distance_scale,
            *args,
            **kwargs
        )
        self.tree: HistoryTree = tree() if tree is not None else None
        self._prune_tree = prune_tree
        self._epoch = 0

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
        env_states = (
            self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states
        )
        # Add corresponding root_walkers data to env_states
        if root_walker is not None:
            if not isinstance(root_walker, OneWalker):
                raise ValueError(
                    "Root walker needs to be an "
                    "instance of OneWalker, got %s instead." % type(root_walker)
                )
            env_states = self._update_env_with_root(root_walker=root_walker, env_states=env_states)

        model_states = (
            self.model.reset(batch_size=len(self.walkers), env_states=env_states)
            if model_states is None
            else model_states
        )
        model_states.update(init_actions=model_states.actions)
        self.walkers.reset(env_states=env_states, model_states=model_states)
        if self.tree is not None:
            root_id = (
                self.walkers.get("id_walkers")[0]
                if root_walker is None
                else copy.copy(root_walker.id_walkers)
            )
            self.tree.reset(
                root_id=root_id,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=self.walkers.states,
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
        """
        Run a new search process.

        Args:
            root_walker: Walker representing the initial state of the search. \
                         The walkers will be reset to this walker, and it will \
                         be added to the root of the :class:`StateTree` if any.
            model_states: :class:`StatesModel` that define the initial state of \
                          the :class:`Model`.
            env_states: :class:`StatesEnv` that define the initial state of \
                        the :class:`Function`.
            walkers_states: :class:`StatesWalkers` that define the internal \
                            states of the :class:`Walkers`.
            report_interval: Display the algorithm progress every ``log_interval`` epochs.
            show_pbar: A progress bar will display the progress of the algorithm run.

        Returns:
            None.

        """
        report_interval = self.report_interval if report_interval is None else report_interval
        self.reset(
            root_walker=root_walker,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
        )

        for _ in self.get_run_loop(show_pbar=show_pbar):
            if self.calculate_end_condition():
                break
            try:
                self.run_step()
                if self.epoch % report_interval == 0 and self.epoch > 0:
                    self.report_progress()
                self.increment_epoch()
            except KeyboardInterrupt:
                break

    def get_run_loop(self, show_pbar: bool = None) -> Iterable[int]:
        """
        Return a tqdm progress bar or a regular range iterator.

        If the code is running in an IPython kernel it will also display the \
        internal ``_notebook_container``.

        Args:
            show_pbar: If ``False`` the progress bar will not be displayed.

        Returns:
            A Progressbar if ``show_pbar`` is ``True`` and the code is running \
            in an IPython kernel. If the code is running in a terminal the logging \
            level must be set at least to "INFO". Otherwise return a range iterator \
            for ``self.max_range`` iteration.

        """
        show_pbar = show_pbar if show_pbar is not None else self.show_pbar
        no_tqdm = not (
            show_pbar if self._ipython_mode else self._log.level < logging.WARNING and show_pbar
        )
        if self._ipython_mode:
            from tqdm.notebook import trange
        else:
            from tqdm import trange

        loop_iterable = trange(
            self.max_epochs, desc="%s" % self.__class__.__name__, disable=no_tqdm
        )

        if self._ipython_mode and self._use_notebook_widget:
            from IPython.core.display import display

            display(self._notebook_container)
        return loop_iterable

    def setup_notebook_container(self):
        """Display the display widgets if the Swarm is running in an IPython kernel."""
        if self._ipython_mode and self._use_notebook_widget:
            from ipywidgets import HTML
            from IPython.core.display import display, HTML as cell_html

            # Set font weight of tqdm progressbar
            display(cell_html("<style> .widget-label {font-weight: bold !important;} </style>"))
            self._notebook_container = HTML()

    def report_progress(self):
        """Report information of the current run."""
        if self._ipython_mode and self._use_notebook_widget:
            line_break = '<br style="line-height:1px; content: "  ";>'
            html = str(self).replace("\n\n", "\n").replace("\n", line_break)
            # Add strong formatting for headers
            html = html.replace("Walkers States", "<strong>Walkers States</strong>")
            html = html.replace("Model States", "<strong>Model States</strong>")
            html = html.replace("Environment States", "<strong>Environment Model</strong>")
            if self.tree is not None:
                tree_name = self.tree.__class__.__name__
                html = html.replace(tree_name, "<strong>%s</strong>" % tree_name)
            self._notebook_container.value = "%s" % html
        elif not self._ipython_mode:
            self._log.info(repr(self))

    def calculate_end_condition(self) -> bool:
        """Implement the logic for deciding if the algorithm has finished. \
        The algorithm will stop if it returns True."""
        return self.walkers.calculate_end_condition()

    def step_and_update_best(self) -> None:
        """
        Make the positions of the walkers evolve and keep track of the new states found.

        It also keeps track of the best state visited.
        """
        self.walkers.update_best()
        self.walkers.fix_best()
        self.step_walkers()

    def balance_and_prune(self) -> None:
        """
        Calculate the virtual reward and perform the cloning process.

        It also updates the :class:`Tree` data structure that takes care of \
        storing the visited states.
        """
        self.walkers.balance()
        self.prune_tree()

    def run_step(self) -> None:
        """
        Compute one iteration of the :class:`Swarm` evolution process and \
        update all the data structures.
        """
        self.step_and_update_best()
        self.balance_and_prune()
        self.walkers.fix_best()

    def step_walkers(self) -> None:
        """
        Make the walkers evolve to their next state sampling an action from the \
        :class:`Model` and applying it to the :class:`Environment`.
        """
        model_states = self.walkers.model_states
        env_states = self.walkers.env_states

        parent_ids = (
            copy.deepcopy(self.walkers.states.id_walkers) if self.tree is not None else None
        )

        model_states = self.model.predict(
            env_states=env_states, model_states=model_states, walkers_states=self.walkers.states
        )
        env_states = self.env.step(model_states=model_states, env_states=env_states)
        self.walkers.update_states(
            env_states=env_states, model_states=model_states,
        )
        self.update_tree(parent_ids)

    def update_tree(self, parent_ids: List[int]) -> None:
        """
        Add a list of walker states represented by `states_ids` to the :class:`Tree`.

        Args:
            parent_ids: list containing the ids of the parents of the new states added.
        """
        if self.tree is not None:
            self.tree.add_states(
                parent_ids=parent_ids,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=self.walkers.states,
                n_iter=int(self.walkers.epoch),
            )

    def prune_tree(self) -> None:
        """
        Remove all the branches that are do not have alive walkers at their leaf nodes.
        """
        if self.tree is not None:
            leaf_nodes = set(self.get("id_walkers"))
            self.tree.prune_tree(alive_leafs=leaf_nodes)

    def _update_env_with_root(self, root_walker, env_states) -> StatesEnv:
        env_states.rewards[:] = copy.deepcopy(root_walker.rewards[0])
        env_states.observs[:] = copy.deepcopy(root_walker.observs[0])
        env_states.states[:] = copy.deepcopy(root_walker.states[0])
        return env_states


class NoBalance(Swarm):
    """Swarm that does not perform the cloning process."""

    def balance_and_prune(self):
        """Do noting."""
        pass

    def calculate_end_condition(self):
        """Finish after reaching the maximum number of epochs."""
        return self.epoch > self.walkers.max_epochs
