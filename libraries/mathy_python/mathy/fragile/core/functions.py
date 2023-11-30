import numpy


def l2_norm(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Euclidean distance between two batches of points stacked across the first dimension."""
    return numpy.linalg.norm(x - y, axis=1)


def relativize(x: numpy.ndarray) -> numpy.ndarray:
    """Normalize the data using a custom smoothing technique."""
    std = x.std()
    if float(std) == 0:
        return numpy.ones(len(x), dtype=type(std))
    standard = (x - x.mean()) / std
    standard[standard > 0] = numpy.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = numpy.exp(standard[standard <= 0])
    return standard
