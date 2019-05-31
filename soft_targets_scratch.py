
import numpy

policy = numpy.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0025706941, 0.0, 0.0, 0.9768637532, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0102827763, 0.0025706941, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0025706941, 0.0025706941, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0025706941, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype="float32",
)
policy_mask = numpy.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype="float32",
)
assert numpy.count_nonzero(policy_mask) == numpy.count_nonzero(policy)
print("policy_mask", policy_mask)
policy_mask *= 0.1
print("policy_mask scaled", policy_mask)
print("policy", policy)
policy_soft = policy + policy_mask
policy_soft /= numpy.sum(policy_soft)
print("policy soft", policy_soft)
assert numpy.count_nonzero(policy_mask) == numpy.count_nonzero(policy)
