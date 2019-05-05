import numpy
import math
from mathy.core.tokenizer import TokenEOF
from mathy.core.expressions import MathTypeKeys

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
TRAIN_LABELS_TARGET_NODE_CONTROL = "node_control"


def parse_example_for_training(example, max_sequence, max_policy_sequence):
    """Prepare a gathered training example for input into the Policy/Value network. 
    This requires padding sequence inputs to the given max length values given as 
    arguments. It returns an output shape that conforms to the structure defined
    by `dataset.make_training_input_fn`
    """
    inputs = {}
    ex_input = example["inputs"]
    # Two extract windows for context sensitivity (3 * 3) = 9
    pad_value = tuple([MathTypeKeys["empty"]] * 9)
    policy_out = example["policy"][:]
    # print(f"Seq={len(ex_input[FEATURE_FWD_VECTORS])}, Policy={len(policy_out)}")

    # Calculate node_control reward value as the absolute value change in the
    # number of context vector floats that are non-zero (i.e. excluding padding)
    last_fwd = numpy.array(ex_input[FEATURE_LAST_FWD_VECTORS]).flatten()
    curr_fwd = numpy.array(ex_input[FEATURE_FWD_VECTORS]).flatten()
    last_seq = len(numpy.trim_zeros(last_fwd))
    curr_seq = len(numpy.trim_zeros(curr_fwd))
    node_ctrl_reward = max(
        0, (max_sequence - int(abs(curr_seq - last_seq))) / max_sequence
    )

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
        ex_input[FEATURE_LAST_BWD_VECTORS][:], max_sequence, pad_value, backwards=True
    )
    policy_out = pad_array(policy_out, max_policy_sequence, 0.0)
    policy_out = numpy.reshape(policy_out, (-1, 6))

    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_BWD_VECTORS])
    inputs[FEATURE_MOVES_REMAINING] = ex_input[FEATURE_MOVES_REMAINING]
    inputs[FEATURE_FOCUS_INDEX] = ex_input[FEATURE_FOCUS_INDEX]
    inputs[FEATURE_MOVE_COUNTER] = ex_input[FEATURE_MOVE_COUNTER]
    inputs[FEATURE_PROBLEM_TYPE] = ex_input[FEATURE_PROBLEM_TYPE]
    # print(inputs[FEATURE_FWD_VECTORS])
    outputs = {
        TRAIN_LABELS_TARGET_PI: policy_out,
        TRAIN_LABELS_TARGET_NODE_CONTROL: [node_ctrl_reward],
        TRAIN_LABELS_TARGET_VALUE: [example["reward"]],
    }
    # print(f"node_control: {outputs[TRAIN_LABELS_TARGET_NODE_CONTROL]}")
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
