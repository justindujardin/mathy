from typing import Callable

import numpy

from fragile.core.states import OneWalker
from fragile.distributed.export_swarm import BestWalker
from fragile.distributed.ray import ray
from fragile.distributed.ray.export_swarm import (
    ExportParamServer as RemoteParamServer,
    ExportSwarm as RemoteExportSwarm,
)


class DistributedExport:
    """
    Run a search process that exchanges :class:`ExportSwarm`.

    This search process uses a :class:`ParamServer` to exchange the specified \
    number o walkers between different :class:`ExportSwarm`.
    """

    def __init__(
        self,
        swarm: Callable,
        n_swarms: 2,
        n_import: int = 2,
        n_export: int = 2,
        export_best: bool = True,
        import_best: bool = True,
        max_len: int = 20,
        add_global_best: bool = True,
        swarm_kwargs: dict = None,
        report_interval: int = numpy.inf,
    ):
        """
        Initialize a :class:`DistributedExport`.

        Args:
            swarm: Callable that returns a :class:`Swarm`. Accepts keyword \
                   arguments defined in ``swarm_kwargs``.
            n_swarms: Number of :class:`ExportSwarm` that will be used in the \
                      to run the search process.
            n_import: Number of walkers that will be imported from an external \
                      :class:`ExportedWalkers`.
            n_export: Number of walkers that will be exported as :class:`ExportedWalkers`.
            export_best: The best walkers of the :class:`Swarm` will always be exported.
            import_best: The best walker of the imported :class:`ExportedWalkers` \
                         will be compared to the best walkers of the \
                         :class:`Swarm`. If it improves the current best value \
                         found, the best walker of the :class:`Swarm` will be updated.
            max_len: Maximum number of :class:`ExportedWalkers` that the \
                     :class:`ParamServer` will keep in its buffer.
            add_global_best: Add the best value found during the search to all \
                             the exported walkers that the :class:`ParamServer` \
                             returns.
            swarm_kwargs: Dictionary containing keyword that will be passed to ``swarm``.
            report_interval: Display the algorithm progress every ``log_interval`` epochs.

        """
        self.report_interval = report_interval
        self.swarms = [
            RemoteExportSwarm.remote(
                swarm=swarm,
                n_export=n_export,
                n_import=n_import,
                import_best=import_best,
                export_best=export_best,
                swarm_kwargs=swarm_kwargs,
            )
            for _ in range(n_swarms)
        ]
        self.n_swarms = n_swarms
        self.minimize = ray.get(self.swarms[0].get.remote("minimize"))
        self.max_epochs = ray.get(self.swarms[0].get.remote("max_epochs"))
        self.reward_limit = ray.get(self.swarms[0].get.remote("reward_limit"))
        self.param_server = RemoteParamServer.remote(
            max_len=max_len, minimize=self.minimize, add_global_best=add_global_best
        )
        self._epoch = 0

    def __getattr__(self, item):
        return ray.get(self.swarms[0].get.remote(item))

    @property
    def epoch(self) -> int:
        """Return the current epoch of the algorithm."""
        return self._epoch

    def get_best(self) -> BestWalker:
        """Return the best walkers found during the algorithm run."""
        return ray.get(self.param_server.get.remote("best"))

    def reset(self, root_walker: OneWalker = None):
        """Reset the internal data of the swarms and parameter server."""
        self._epoch = 0
        reset_param_server = self.param_server.reset.remote()
        reset_swarms = [swarm.reset.remote(root_walker=root_walker) for swarm in self.swarms]
        ray.get(reset_param_server)
        ray.get(reset_swarms)

    def run(self, root_walker: OneWalker = None, report_interval=None):
        """
        Run the distributed search algorithm asynchronously.

        Args:
            root_walker: Walker representing the initial state of the search. \
                         The walkers will be reset to this walker, and it will \
                         be added to the root of the :class:`StateTree` if any.
            report_interval: Display the algorithm progress every ``report_interval`` epochs.

        """
        report_interval = self.report_interval if report_interval is None else report_interval
        self.reset(root_walker=root_walker)
        current_import_walkers = self.swarms[0].get_empty_export_walkers.remote()
        steps = {}
        for swarm in self.swarms:
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

        for i in range(self.max_epochs * self.n_swarms):
            self._epoch = i // self.n_swarms
            ready_export_walkers, _ = ray.wait(list(steps))
            ready_export_walker_id = ready_export_walkers[0]
            swarm = steps.pop(ready_export_walker_id)

            # Compute and apply gradients.
            current_import_walkers = self.param_server.exchange_walkers.remote(
                ready_export_walker_id
            )
            steps[swarm.run_exchange_step.remote(current_import_walkers)] = swarm

            if self.epoch % report_interval == 0 and self.epoch > 0:
                # Evaluate the current model after every 10 updates.
                best = self.get_best()
                print("iter {} best_reward: {:.3f}".format(i, best.rewards))
