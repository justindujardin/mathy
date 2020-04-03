import pytest
import random
from mathy.problems import rand_number, use_pretty_numbers


def test_number_generation() -> None:

    random.seed(1337)

    # When using pretty number generation, all values
    # are in range 1-12 and are always integers
    use_pretty_numbers(True)
    pretty_numbers = [rand_number() for _ in range(256)]
    outside_range_floats = [f for f in pretty_numbers if f < 1 or f > 12]
    pretty_floats = [f for f in pretty_numbers if isinstance(f, float)]
    assert len(outside_range_floats) == 0
    assert len(pretty_floats) == 0

    # When not using pretty numbers, values can be floats and large integers
    use_pretty_numbers(False)
    rand_numbers = [rand_number() for _ in range(256)]
    large_ints = [f for f in rand_numbers if isinstance(f, int) and f > 12]
    rand_floats = [f for f in rand_numbers if isinstance(f, float)]
    assert len(large_ints) > 0
    assert len(rand_floats) > 0
