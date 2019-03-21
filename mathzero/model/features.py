import numpy

FEATURE_TOKEN_VALUES = "token_values"
FEATURE_TOKEN_TYPES = "token_types"
FEATURE_NODE_COUNT = "node_count"
FEATURE_MOVES_REMAINING = "moves_remaining"
FEATURE_MOVE_COUNTER = "move_counter"
FEATURE_PROBLEM_TYPE = "problem_type"
FEATURE_COLUMNS = [
    FEATURE_TOKEN_VALUES,
    FEATURE_TOKEN_TYPES,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    FEATURE_MOVES_REMAINING,
    FEATURE_MOVE_COUNTER,
]


TRAIN_LABELS_TARGET_PI = "policy"
TRAIN_LABELS_TARGET_VALUE = "value"
TRAIN_LABELS_AS_MATRIX = "matrix"


def parse_example_for_training(example, max_sequence=None):
    """Wrapper that accepts a single example to parse for feeding to the network.
    
    Returns: a tuple of(features, labels) """
    inputs = {}
    ex_input = example["inputs"]
    for feature_key in FEATURE_COLUMNS:
        inputs[feature_key] = ex_input[feature_key]
    if max_sequence is not None:
        inputs[FEATURE_TOKEN_TYPES] = pad_array(
            inputs[FEATURE_TOKEN_TYPES], max_sequence
        )
        inputs[FEATURE_TOKEN_VALUES] = pad_array(
            inputs[FEATURE_TOKEN_VALUES], max_sequence
        )
    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_TOKEN_TYPES])
    outputs = {
        TRAIN_LABELS_TARGET_PI: example["policy"],
        TRAIN_LABELS_TARGET_VALUE: [example["reward"]],
    }
    return inputs, outputs


def pad_array(A, max_length, value=0):
    """Pad a numpy array to the given size with the given padding value"""
    a_len = len(A)
    if a_len >= max_length:
        return A
    t = max_length - len(A)
    return numpy.pad(A, pad_width=(0, t), mode="constant", constant_values=value)
