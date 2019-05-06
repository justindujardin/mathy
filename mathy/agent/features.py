import numpy
import math
from mathy.core.tokenizer import TokenEOF
from mathy.core.expressions import MathTypeKeys

FEATURE_FWD_VECTORS = "fwd_vectors"
FEATURE_LAST_RULE = "last_action"
FEATURE_BWD_VECTORS = "bwd_vectors"
FEATURE_LAST_FWD_VECTORS = "fwd_last_vectors"
FEATURE_LAST_BWD_VECTORS = "bwd_last_vectors"
FEATURE_NODE_COUNT = "node_count"
FEATURE_MOVES_REMAINING = "moves_remaining"
FEATURE_MOVE_COUNTER = "move_counter"
FEATURE_PROBLEM_TYPE = "problem_type"


TRAIN_LABELS_TARGET_PI = "policy"
TRAIN_LABELS_TARGET_VALUE = "value"
TRAIN_LABELS_TARGET_NODE_CONTROL = "node_ctrl"


def calculate_node_control_signal(example_inputs, max_sequence):
    """Calculate node_ctrl signal as the absolute value change in the
    number of context vector floats that are non-zero (i.e. excluding padding)
    in the expression.
    """
    last_fwd = numpy.array(example_inputs[FEATURE_LAST_FWD_VECTORS]).flatten()
    curr_fwd = numpy.array(example_inputs[FEATURE_FWD_VECTORS]).flatten()
    last_seq = len(numpy.trim_zeros(last_fwd))
    curr_seq = len(numpy.trim_zeros(curr_fwd))
    node_ctrl_loss = max(
        0, (max_sequence - int(abs(curr_seq - last_seq))) / max_sequence
    )
    return node_ctrl_loss


def calculate_grouping_control_signal(example_inputs, max_sequence):
    """Calculate grouping_control signals as the sum of all distances between 
    all like terms. Iterate over each context vector and extract the node type
    it represents
     
    number of context vector floats that are non-zero (i.e. excluding padding)
    """
    raise EnvironmentError("unimplemented")


def parse_example_for_training(example, max_sequence, max_policy_sequence):
    """Prepare a gathered training example for input into the Policy/Value network. 
    This requires padding sequence inputs to the given max length values given as 
    arguments. It returns an output shape that conforms to the structure defined
    by `dataset.make_training_input_fn`
    """
    inputs = {}
    ex_input = example["features"]
    num_actions = (
        6
    )  # TODO: This is hardcoded to the number of rules in math_game.py FIXIT!
    # Two extract windows for context sensitivity (3 * 3) = 9
    pad_value = tuple([MathTypeKeys["empty"]] * 9)
    policy_out = numpy.array(example["policy"][:]).flatten().tolist()
    # print(f"Seq={len(ex_input[FEATURE_FWD_VECTORS])}, Policy={len(policy_out)}")

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
    policy_out = numpy.reshape(policy_out, (-1, num_actions))

    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_BWD_VECTORS])
    inputs[FEATURE_MOVES_REMAINING] = ex_input[FEATURE_MOVES_REMAINING]
    inputs[FEATURE_LAST_RULE] = ex_input[FEATURE_LAST_RULE]
    inputs[FEATURE_MOVE_COUNTER] = ex_input[FEATURE_MOVE_COUNTER]
    inputs[FEATURE_PROBLEM_TYPE] = ex_input[FEATURE_PROBLEM_TYPE]
    # print(inputs[FEATURE_FWD_VECTORS])
    outputs = {
        TRAIN_LABELS_TARGET_PI: policy_out,
        TRAIN_LABELS_TARGET_NODE_CONTROL: [
            calculate_node_control_signal(ex_input, max_sequence)
        ],
        TRAIN_LABELS_TARGET_VALUE: [example["discounted"]],
    }
    # print(f"node_ctrl: {outputs[TRAIN_LABELS_TARGET_NODE_CONTROL]}")
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
