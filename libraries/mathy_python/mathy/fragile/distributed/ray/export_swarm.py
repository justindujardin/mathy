from typing import Callable

from fragile.distributed.export_swarm import (
    ExportedWalkers,
    ExportSwarm as WorkerExportSwarm,
    ParamServer,
)
from fragile.distributed.ray import ray


@ray.remote
class ExportParamServer(ParamServer):
    """:class:`ParamServer` that can be used with ray."""

    def get(self, name: str, default=None):
        """Access attributes of :class:`ParamServer`."""
        try:
            return getattr(self, name)
        except Exception:
            return default


@ray.remote
class ExportSwarm:
    """Allows to use a :class:`ExportSwarm` with ray."""

    def __init__(
        self,
        swarm: Callable,
        n_import: int = 2,
        n_export: int = 2,
        export_best: bool = True,
        import_best: bool = True,
        swarm_kwargs: dict = None,
    ):
        """
        Initialize a :class:`RemoteExportsWarm`.

        Args:
            swarm: Callable that returns a :class:`Swarm`. Accepts keyword \
                   arguments defined in ``swarm_kwargs``.
            n_import: Number of walkers that will be imported from an external \
                      :class:`ExportedWalkers`.
            n_export: Number of walkers that will be exported as :class:`ExportedWalkers`.
            export_best: The best walkers of the :class:`Swarm` will always be exported.
            import_best: The best walker of the imported :class:`ExportedWalkers` \
                         will be compared to the best walkers of the \
                         :class:`Swarm`. If it improves the current best value \
                         found, the best walker of the :class:`Swarm` will be updated.
            swarm_kwargs: Dictionary containing keyword that will be passed to ``swarm``.
        """
        swarm_kwargs = swarm_kwargs if swarm_kwargs is not None else {}
        swarm = swarm(**swarm_kwargs)
        self.swarm = WorkerExportSwarm(
            swarm=swarm,
            n_export=n_export,
            n_import=n_import,
            import_best=import_best,
            export_best=export_best,
        )

    def reset(self, *args, **kwargs):
        """Reset the internal :class:`ExportSwarm`."""
        self.swarm.reset(*args, **kwargs)

    # Ray does not allow to implement static methods in remote classes.
    def get_empty_export_walkers(self) -> ExportedWalkers:
        """
        Return a :class:`ExportedWalkers` with no walkers inside.

        Used to initialize the algorithm.
        """
        return ExportedWalkers(0)

    def run_exchange_step(self, walkers: ExportedWalkers) -> ExportedWalkers:
        """Run a the walkers import/export process of the internal :class:`ExportSwarm`."""
        return self.swarm.run_exchange_step(walkers)

    def get(self, name: str):
        """Access attributes of the underlying :class:`ExportSwarm`."""
        if hasattr(self.swarm.walkers.states, name):
            return getattr(self.swarm.walkers.states, name)
        elif hasattr(self.swarm.walkers.env_states, name):
            return getattr(self.swarm.walkers.env_states, name)
        elif hasattr(self.swarm.walkers.model_states, name):
            return getattr(self.swarm.walkers.model_states, name)
        elif hasattr(self.swarm.walkers, name):
            return getattr(self.swarm.walkers, name)
        elif hasattr(self.swarm, name):
            return getattr(self.swarm, name)
        else:
            raise ValueError("%s is not an attribute of the states, swarm or walkers." % name)
