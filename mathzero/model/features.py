import numpy
from mathzero.core.tokenizer import TokenEOF

FEATURE_TOKEN_VALUES = "token_values"
FEATURE_TOKEN_TYPES = "token_types"
FEATURE_LAST_TOKEN_VALUES = "last_token_values"
FEATURE_LAST_TOKEN_TYPES = "last_token_types"
FEATURE_NODE_COUNT = "node_count"
FEATURE_MOVES_REMAINING = "moves_remaining"
FEATURE_MOVE_COUNTER = "move_counter"
FEATURE_PROBLEM_TYPE = "problem_type"


TRAIN_LABELS_TARGET_PI = "policy"
TRAIN_LABELS_TARGET_VALUE = "value"
TRAIN_LABELS_AS_MATRIX = "matrix"


def parse_example_for_training(example, max_sequence=None):
    """Wrapper that accepts a single example to parse for feeding to the network.
    
    Returns: a tuple of(features, labels) """
    inputs = {}
    ex_input = example["inputs"]
    if max_sequence is not None:
        # inputs[FEATURE_TOKEN_TYPES] = pad_array(
        #     ex_input[FEATURE_TOKEN_TYPES], max_sequence, TokenEOF
        # )
        inputs[FEATURE_TOKEN_VALUES] = pad_array(
            ex_input[FEATURE_TOKEN_VALUES], max_sequence, ""
        )
        # inputs[FEATURE_LAST_TOKEN_TYPES] = pad_array(
        #     ex_input[FEATURE_LAST_TOKEN_TYPES], max_sequence, TokenEOF
        # )
        # inputs[FEATURE_LAST_TOKEN_VALUES] = pad_array(
        #     ex_input[FEATURE_LAST_TOKEN_VALUES], max_sequence, ""
        # )
    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_TOKEN_TYPES])
    inputs[FEATURE_MOVES_REMAINING] = ex_input[FEATURE_MOVES_REMAINING]
    inputs[FEATURE_MOVE_COUNTER] = ex_input[FEATURE_MOVE_COUNTER]
    inputs[FEATURE_PROBLEM_TYPE] = ex_input[FEATURE_PROBLEM_TYPE]
    outputs = {
        TRAIN_LABELS_TARGET_PI: example["policy"],
        TRAIN_LABELS_TARGET_VALUE: [example["reward"]],
    }
    return inputs, outputs


def pad_array(A, max_length, value=0):
    """Pad a numpy array to the given size with the given padding value"""
    while len(A) < max_length:
        A.append(value)
    return A
