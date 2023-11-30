from typing import Callable, Iterable, List, Tuple

import numpy
import ray
from tqdm.auto import tqdm

from fragile.core.base_classes import BaseEnvironment, BaseModel
from fragile.core.memory import ReplayMemory as CoreMemory
from fragile.core.states import OneWalker
from fragile.core.tree import HistoryTree, NamesData, NetworkxTree, NodeId
from fragile.core.walkers import Walkers
from fragile.distributed.ray.swarm import RemoteSwarm


@ray.remote
class ReplayMemory(CoreMemory):
    """Remote replay memory."""


class ReplayCreator:
    """
    Generate replay data using several :class:`Swarm` in parallel and store it \
     in a :class:`ReplayMemory`.

    Every :class:`Swarm` will use the specified number of workers to step \
    the environment.
    """

    def __init__(
        self,
        n_swarms: int,
        n_workers_per_swarm: int,
        names: List,
        num_examples: int,
        n_walkers: int,
        env: Callable[[], BaseEnvironment],
        model: Callable[[BaseEnvironment], BaseModel],
        max_examples: int = None,
        mode: str = "best",
        walkers: Callable[..., Walkers] = Walkers,
        reward_scale: float = 1.0,
        distance_scale: float = 1.0,
        report_interval: int = numpy.inf,
        show_pbar: bool = False,
        use_notebook_widget: bool = True,
        force_logging: bool = False,
        prune_tree: bool = True,
        root_id: NodeId = 0,
        node_names: NamesData = NetworkxTree.DEFAULT_NODE_DATA,
        edge_names: NamesData = NetworkxTree.DEFAULT_EDGE_DATA,
        next_prefix: str = NetworkxTree.DEFAULT_NEXT_PREFIX,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`Swarm`.

        Args:
            n_walkers: Number of walkers of the swarm.
            n_swarms: Number of swarms that will be run in parallel to generate replay data.
            n_workers_per_swarm: Number of worker processes spawned by the remote environment.
            env: A callable that returns an instance of an Environment.
            model: A callable that returns an instance of a Model.
            walkers: A callable that returns an instance of BaseWalkers.
            reward_scale: Virtual reward exponent for the reward score.
            distance_scale: Virtual reward exponent for the distance score.
            names: Names of the data attributes that will be extracted stored \
                   in the graph data. The data generators will return the data
                   as a tuple of arrays, ordered according to ``names``.
            max_examples: Maximum number of experiences that will be stored.
            mode: If ``mode == "best"`` store only data from the best trajectory \
                  of the :class:`Swarm`. Otherwise store data from all the states of \
                  the :class:`HistoryTree`.

            num_examples: Minimum number of samples that need to be stored before the \
                     replay memory is considered ready. If ``None`` it will be equal \
                     to max_size.
            prune_tree: If ``True`` the tree will be pruned after every iteration to \
                  remove the branches that have stopped its expansion process.
            root_id: The node id of the root node.
            next_prefix: Prefix used to refer to data extracted from the next \
                        node when parsing a data generator. For example: \
                        "next_observs" will reference the observation of the \
                        next node.
            node_names: Names of the data attributes of the :class:`States` that \
                       will be stored as node attributes in the internal graph.
            edge_names: Names of the data attributes of the :class:`States` that \
                       will be stored as edge attributes in the internal graph.
            report_interval: Ignored. Only used to match swarm interface.
            show_pbar: If ``True`` A progress bar will display the progress of \
                       the algorithm run.
            use_notebook_widget: If ``True`` and the class is running in an IPython \
                                kernel it will display the evolution of the swarm \
                                in a widget.
            force_logging: If ``True``, disable al ``ipython`` related behaviour.
            *args: Additional args passed to init_swarm.
            **kwargs: Additional kwargs passed to init_swarm.

        """

        def tree_callable():
            return HistoryTree(
                prune=prune_tree,
                names=names,
                root_id=root_id,
                node_names=node_names,
                edge_names=edge_names,
                next_prefix=next_prefix,
            )

        self.swarms = [
            RemoteSwarm.remote(
                n_walkers=n_walkers,
                env=env,
                model=model,
                walkers=walkers,
                reward_scale=reward_scale,
                distance_scale=distance_scale,
                tree=tree_callable,
                report_interval=report_interval,
                show_pbar=False,
                use_notebook_widget=False,
                force_logging=True,
                n_workers=n_workers_per_swarm,
                *args,
                **kwargs
            )
            for _ in range(n_swarms)
        ]
        max_size = max_examples if max_examples is not None else num_examples
        self.memory = ReplayMemory.remote(
            max_size=max_size, names=names, mode=mode, min_size=num_examples
        )
        self._names = names
        self.memory_length = 0
        self.target_memory_size = ray.get(self.memory.get.remote("min_size"))
        self.show_pbar = show_pbar
        self.force_loggin = force_logging
        self.use_notebook_widget = use_notebook_widget
        self.max_epochs = ray.get(self.swarms[0].get.remote("max_epochs"))

    @property
    def names(self):
        """Return the data attributes of each state that will be stored."""
        return self._names

    def __getattr__(self, item):
        if item in self.names:
            return ray.get(self.memory.get.remote(item))
        return self.__getattribute__(item)

    def reset(self, root_walker: OneWalker = None):
        """Reset the internal data of the swarms and parameter server."""
        reset_memory = self.memory.reset.remote()
        self.memory_length = 0
        reset_swarms = [swarm.reset.remote(root_walker=root_walker) for swarm in self.swarms]
        ray.get([reset_memory] + reset_swarms)

    def get_memory_length(self) -> int:
        """Return the length of the remote memory being use to store replay data."""
        mem_len = ray.get(self.memory.get.remote("len"))
        return mem_len

    def get_values(self) -> Tuple[numpy.ndarray, ...]:
        """Return a tuple containing the data attributes stored in the :class:`ReplayMemory`."""
        return ray.get(self.memory.get_values.remote())

    def iterate_values(self) -> Iterable[Tuple[numpy.ndarray]]:
        """
        Return a generator that yields a tuple containing the data of each state \
        stored in the memory.
        """
        return ray.get(self.memory.iterate_values.remote())

    def run(self, root_walker: OneWalker = None, report_interval=None):
        """
        Run the distributed search algorithm asynchronously.

        Args:
            root_walker: Walker representing the initial state of the search. \
                         The walkers will be reset to this walker, and it will \
                         be added to the root of the :class:`StateTree` if any.
            report_interval: Ignored.

        """
        self.reset(root_walker=root_walker)
        swarm_runs = {}
        for swarm in self.swarms:
            swarm_runs[swarm.run.remote()] = swarm

        memorize_id = None
        with tqdm(
            total=self.target_memory_size, disable=not self.show_pbar, desc="Generated examples"
        ) as pbar:
            while self.memory_length < self.target_memory_size:
                if memorize_id is not None:  # Update remote memory and pbar progress
                    ray.get(memorize_id)
                    new_mem_len = self.get_memory_length()
                    pbar_update = new_mem_len - self.memory_length
                    self.memory_length = new_mem_len
                    if self.memory_length > self.target_memory_size:
                        pbar.total = self.memory_length
                    pbar.update(pbar_update)
                # Send update to memory an run swarm again
                ready_swarms, _ = ray.wait(list(swarm_runs))
                ready_swarm_id = ready_swarms[0]
                swarm = swarm_runs.pop(ready_swarm_id)
                memorize_id = self.memory.memorize.remote(ready_swarm_id)
                swarm_runs[swarm.run.remote()] = swarm
