from typing import Any, Callable, Dict, Generator, Tuple, Union

import numpy
import xxhash


RANDOM_SEED = 160290
random_state = numpy.random.RandomState(seed=RANDOM_SEED)

hash_type = "<U64"
float_type = numpy.float32
Scalar = Union[int, int, float, float]
StateDict = Dict[str, Dict[str, Any]]
DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]

NUMPY_IGNORE_WARNINGS_PARAMS = {
    "divide": "ignore",
    "over": "ignore",
    "under": "ignore",
    "invalid": "ignore",
}


def running_in_ipython() -> bool:
    """Return ``True`` if the code is this function is being called from an IPython kernel."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def hash_numpy(x: numpy.ndarray) -> int:
    """Return a value that uniquely identifies a numpy array."""
    return xxhash.xxh64_hexdigest(x.tobytes())


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return x.mean(), x.std(), x.max(), x.min()
    except AttributeError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


def similiar_chunks_indexes(
    n_values, n_chunks
) -> Generator[Tuple[int, int], None, None]:
    """
    Return the indexes for splitting an array in similar chunks.

    Args:
        n_values: Length of the array that will be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator containing the indexes of every new chunk.

    """
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for i in range(0, n_values, chunk_size):
        yield i, i + chunk_size


def split_kwargs_in_chunks(kwargs, n_chunks):
    """Split the kwargs passed to ``make_transitions`` in similar batches."""
    n_values = len(next(iter(kwargs.values())))  # Assumes all data have the same len
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for start, end in similiar_chunks_indexes(n_values, n_chunks):
        if (
            start + chunk_size >= n_values - 2
        ):  # Do not allow the last chunk to have size 1
            yield {
                k: v[start:n_values] if isinstance(v, numpy.ndarray) else v
                for k, v in kwargs.items()
            }
            break
        else:
            yield {
                k: v[start:end] if isinstance(v, numpy.ndarray) else v
                for k, v in kwargs.items()
            }


def split_args_in_chunks(args, n_chunks):
    """Split the args passed to ``make_transitions`` in similar batches."""
    n_values = len(args[0])
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for start, end in similiar_chunks_indexes(n_values, n_chunks):
        if start + chunk_size >= n_values - 2:
            yield tuple(v[start:n_values] for v in args)
            break
        yield tuple(v[start:end] for v in args)
