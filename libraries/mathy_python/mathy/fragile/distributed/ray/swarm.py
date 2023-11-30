import copy
from typing import Callable

import ray

from fragile.core.base_classes import BaseEnvironment, BaseModel, BaseTree
from fragile.core.states import OneWalker, StatesEnv, StatesModel, StatesWalkers
from fragile.core.swarm import Swarm as CoreSwarm, Walkers as CoreWalkers
from fragile.distributed.ray.env import RayEnv


@ray.remote
class RemoteSwarm(CoreSwarm):
    """
    Swarm that runs inside a ``ray`` worker process.

    It uses a remote :class:`Environment` to step the walkers in parallel.
    """

    def init_swarm(
        self,
        env_callable: Callable[[], BaseEnvironment],
        model_callable: Callable[[BaseEnvironment], BaseModel],
        walkers_callable: Callable[..., CoreWalkers],
        n_walkers: int,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        tree: Callable[[], BaseTree] = None,
        prune_tree: bool = True,
        n_workers: int = 8,
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
            n_workers: Number of worker processes spawned by the remote environment.
            args: Passed to ``walkers_callable``.
            kwargs: Passed to ``walkers_callable``.

        Returns:
            None.

        """
        # Remote classes raise error when calling super()__init__(), so the class attributes are
        # defined in this function.
        self.n_workers = n_workers
        self._env = RayEnv.remote(env_callable=env_callable, n_workers=n_workers)
        local_env = env_callable()
        env_params = local_env.get_params_dict()
        self._model = model_callable(local_env)
        model_params = self._model.get_params_dict()
        self._walkers: CoreWalkers = CoreWalkers(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            distance_scale=distance_scale,
            *args,
            **kwargs
        )
        self.tree = tree() if tree is not None else None
        self._prune_tree = prune_tree
        self._epoch = 0

    async def reset(
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
        n_walkers = self.walkers.get("n_walkers")
        reset_id = (
            self.env.reset.remote(batch_size=n_walkers) if env_states is None else env_states
        )
        env_states = await reset_id
        # Add corresponding root_walkers data to env_states
        if root_walker is not None:
            if not isinstance(root_walker, OneWalker):
                raise ValueError(
                    "Root walker needs to be an "
                    "instance of OneWalker, got %s instead." % type(root_walker)
                )
            env_states = self._update_env_with_root(root_walker=root_walker, env_states=env_states)

        model_states = (
            self.model.reset(batch_size=n_walkers, env_states=env_states)
            if model_states is None
            else model_states
        )
        model_states.update(init_actions=model_states.actions)
        self.walkers.reset(env_states=env_states, model_states=model_states)
        if self.tree is not None:
            id_walkers = self.walkers.get("id_walkers")
            root_id = id_walkers[0] if root_walker is None else copy.copy(root_walker.id_walkers)
            self.tree.reset(
                root_id=root_id,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=self.walkers.states,
            )

    def calculate_end_condition(self) -> bool:
        """Implement the logic for deciding if the algorithm has finished. \
        The algorithm will stop if it returns True."""
        return self.walkers.calculate_end_condition()

    def step_and_update_best(self):
        """
        Make the positions of the walkers evolve and keep track of the new states found.

        It also keeps track of the best state visited.
        """
        self.walkers.update_best()
        self.walkers.fix_best()
        coroutine = self.step_walkers()
        return coroutine

    def balance_and_prune(self) -> None:
        """
        Calculate the virtual reward and perform the cloning process.

        It also updates the :class:`Tree` data structure that takes care of \
        storing the visited states.
        """
        self.walkers.balance()
        self.prune_tree()

    async def run_step(self) -> None:
        """
        Compute one iteration of the :class:`Swarm` evolution process and \
        update all the data structures.
        """
        await self.step_and_update_best()
        self.balance_and_prune()
        self.walkers.fix_best()

    async def step_walkers(self) -> None:
        """
        Make the walkers evolve to their next state sampling an action from the \
        :class:`Model` and applying it to the :class:`Environment`.
        """
        model_states = self.walkers.get("model_states")
        env_states = self.walkers.get("env_states")
        walkers_states = self.walkers.get("states")
        parent_ids = (
            copy.deepcopy(self.walkers.get("id_walkers")) if self.tree is not None else None
        )

        model_states = self.model.predict(
            env_states=env_states, model_states=model_states, walkers_states=walkers_states
        )
        env_states = await self.env.step.remote(model_states=model_states, env_states=env_states)
        # env_states = ray.get(step_id)
        self.walkers.update_states(
            env_states=env_states, model_states=model_states,
        )
        self.update_tree(parent_ids)

    def increment_epoch(self) -> None:
        """Increment the current epoch of the algorithm."""
        self._epoch += 1
        self.walkers.increment_epoch()

    async def run(
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
        await self.reset(
            root_walker=root_walker,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
        )
        for _ in self.get_run_loop(show_pbar=show_pbar):
            if self.calculate_end_condition():
                break
            try:
                await self.run_step()
                if self.epoch % report_interval == 0 and self.epoch > 0:
                    self.report_progress()
                self.increment_epoch()
            except KeyboardInterrupt:
                break
        return self
