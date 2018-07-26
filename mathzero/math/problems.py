def random_ints_with_sum(n):
    """
    Generate non-negative random integers summing to `n`.
    """
    while n > 0:
        r = random.randint(0, n)
        yield r
        n -= r


class ProblemGenerator:
    