import numpy
from mathzero.core.tokenizer import TokenEOF
from mathzero.core.expressions import MathTypeKeys

FEATURE_FWD_VECTORS = "fwd_vectors"
FEATURE_FOCUS_INDEX = "focus_index"
FEATURE_BWD_VECTORS = "bwd_vectors"
FEATURE_LAST_FWD_VECTORS = "fwd_last_vectors"
FEATURE_LAST_BWD_VECTORS = "bwd_last_vectors"
FEATURE_NODE_COUNT = "node_count"
FEATURE_MOVES_REMAINING = "moves_remaining"
FEATURE_MOVE_COUNTER = "move_counter"
FEATURE_PROBLEM_TYPE = "problem_type"


TRAIN_LABELS_TARGET_PI = "policy"
TRAIN_LABELS_TARGET_VALUE = "value"
TRAIN_LABELS_AS_MATRIX = "matrix"


def parse_example_for_training(example, max_sequence=None, max_policy_sequence=None):
    """Wrapper that accepts a single example to parse for feeding to the network.
    
    Returns: a tuple of(features, labels) """
    inputs = {}
    ex_input = example["inputs"]
    # Two extract windows for context sensitivity (3 * 3) = 9
    pad_value = (
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
        MathTypeKeys["empty"],
    )
    policy_out = example["policy"][:]
    # print(f"Seq={len(ex_input[FEATURE_FWD_VECTORS])}, Policy={len(policy_out)}")
    if max_sequence is not None:
        inputs[FEATURE_FWD_VECTORS] = pad_array(
            ex_input[FEATURE_FWD_VECTORS][:], max_sequence, pad_value
        )
        inputs[FEATURE_BWD_VECTORS] = pad_array(
            ex_input[FEATURE_BWD_VECTORS][:], max_sequence, pad_value, backwards=True
        )
        inputs[FEATURE_LAST_FWD_VECTORS] = pad_array(
            ex_input[FEATURE_LAST_FWD_VECTORS][:], max_sequence, pad_value
        )
        inputs[FEATURE_LAST_BWD_VECTORS] = pad_array(
            ex_input[FEATURE_LAST_BWD_VECTORS][:],
            max_sequence,
            pad_value,
            backwards=True,
        )
    if max_policy_sequence is not None:
        policy_out = pad_array(policy_out, max_policy_sequence, 0.0)
        policy_out = [[p] for p in policy_out]

    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_BWD_VECTORS])
    inputs[FEATURE_MOVES_REMAINING] = ex_input[FEATURE_MOVES_REMAINING]
    inputs[FEATURE_FOCUS_INDEX] = ex_input[FEATURE_FOCUS_INDEX]
    inputs[FEATURE_MOVE_COUNTER] = ex_input[FEATURE_MOVE_COUNTER]
    inputs[FEATURE_PROBLEM_TYPE] = ex_input[FEATURE_PROBLEM_TYPE]
    # print(inputs[FEATURE_FWD_VECTORS])
    outputs = {
        TRAIN_LABELS_TARGET_PI: policy_out,
        TRAIN_LABELS_TARGET_VALUE: [example["reward"]],
    }
    return inputs, outputs


def pad_array(A, max_length, value=0, backwards=False):
    """Pad a list to the given size with the given padding value
    
    If backwards=True the input will be reversed after padding, and 
    the output will be reversed after padding, to correctly pad for 
    LSTMs, e.g. "4x+2----" padded backwards would be "----2+x4"
    """
    if backwards:
        A.reverse()
    while len(A) < max_length:
        A.append(value)
    if backwards:
        A.reverse()
    return A
