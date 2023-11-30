import logging
from typing import Iterable, List, Tuple, Union

import numpy

from fragile.core.swarm import Swarm


class ReplayMemory:
    """Store replay data extracted from a :class:`HistoryTree`."""

    _log = logging.getLogger("Replaymemory")

    def __init__(
        self,
        max_size: int,
        names: Union[List[str], Tuple[str]],
        mode: str = "best",
        min_size: int = None,
    ):
        """
        Initialize a :class:`ReplayMemory`.

        Args:
            max_size: Maximum number of experiences that will be stored.
            names: Names of the replay data attributes that will be stored.
            mode: If ``mode == "best"`` store only data from the best trajectory \
                  of the :class:`Swarm`. Otherwise store data from all the states of \
                  the :class:`HistoryTree`.

            min_size: Minimum number of samples that need to be stored before the \
                     replay memory is considered ready. If ``None`` it will be equal \
                     to max_size.

        """
        self.max_size = max_size
        self.min_size = max_size if min_size is None else min_size
        self.names = names
        self.mode = mode
        self.reset()

    def __len__(self):
        if getattr(self, self.names[0]) is None:
            return 0
        return len(getattr(self, self.names[0]))

    def __repr__(self) -> str:
        text = "Memory with min_size %s max_size %s and length %s" % (
            self.min_size,
            self.max_size,
            len(self),
        )
        return text

    def reset(self):
        """Delete all the data previously stored in the memory."""
        for name in self.names:
            setattr(self, name, None)

    def get(self, name):
        """Get attributes of the memory."""
        if name == "len":
            return len(self)
        return getattr(self, name)

    def is_ready(self) -> bool:
        """
        Return ``True`` if the number of experiences in the memory is greater than ``min_size``.
        """
        return len(self) >= self.min_size

    def memorize(self, swarm: Swarm):
        """
        Extract the replay data from a :class:`Swarm` and incorporate it to the \
        already saved experiences.
        """
        # extract data from the swarm
        if self.mode == "best":
            data = next(swarm.tree.iterate_branch(swarm.best_id, batch_size=-1, names=self.names))
        else:
            data = next(swarm.tree.iterate_nodes_at_random(batch_size=-1, names=self.names))
        # Concatenate the data to the current memory
        for name, val in zip(self.names, data):
            # Scalar vectors are transformed to columns
            if isinstance(val, numpy.ndarray) and len(val.shape) == 1:
                val = val.reshape(-1, 1)
            processed = (
                val if getattr(self, name) is None else numpy.vstack([val, getattr(self, name)])
            )
            if len(processed) > self.max_size:
                processed = processed[: self.max_size]
            setattr(self, name, processed)
        self._log.info("Memory now contains %s samples" % len(self))

    def get_values(self) -> Tuple[numpy.ndarray, ...]:
        """Return a tuple containing the memorized data for all the saved data attributes."""
        return tuple([getattr(self, val) for val in self.names])

    def iterate_values(self) -> Iterable[Tuple[numpy.ndarray]]:
        """
        Return a generator that yields a tuple containing the data of each state \
        stored in the memory.
        """
        for i in range(len(self)):
            yield tuple([getattr(self, val)[i] for val in self.names])
