import numpy
import logging
from json import loads, dumps
from itertools import zip_longest

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


def lengths(x):
    if isinstance(x, list):
        yield len(x)
        for y in x:
            yield from lengths(y)


def parse_examples_for_training(examples):
    """Parse a given JSONL dataset of examples into an x/y output
    params:
        examples: the JSONL items from `examples.jsonl`
    returns: 
        tuple of (examples, labels) for training where labels are a tuple 
        of (target_policy, target_reward, target_focus)
    """
    import tensorflow as tf

    inputs = {}
    outputs = []
    for feature_key in FEATURE_COLUMNS:
        inputs[feature_key] = []
    # Build up a feature map that can work as input
    for ex in examples:
        ex_input = ex["inputs"]
        ex_append = {}
        for feature_key in FEATURE_COLUMNS:
            inputs[feature_key].append(ex_input[feature_key])
        target_pi = numpy.array(ex["policy"], dtype="float32")
        target_reward = ex["reward"]
        target_focus = ex["focus"]
        outputs.append(
            numpy.concatenate((target_pi, [target_reward, target_focus]), axis=0)
        )

    # Pad the variable length columns to longest in the list
    max_sequence = max(lengths(inputs[FEATURE_TOKEN_TYPES]))
    inputs[FEATURE_TOKEN_TYPES] = numpy.array(
        list(zip_longest(*inputs[FEATURE_TOKEN_TYPES], fillvalue=0)), dtype="int32"
    ).T
    inputs[FEATURE_TOKEN_VALUES] = numpy.array(
        list(zip_longest(*inputs[FEATURE_TOKEN_VALUES], fillvalue=0))
    ).T
    inputs[FEATURE_NODE_COUNT] = numpy.array(inputs[FEATURE_NODE_COUNT], dtype="int16")
    inputs[FEATURE_MOVES_REMAINING] = numpy.array(
        inputs[FEATURE_MOVES_REMAINING], dtype="int16"
    )
    inputs[FEATURE_MOVE_COUNTER] = numpy.array(
        inputs[FEATURE_MOVE_COUNTER], dtype="int16"
    )
    inputs[FEATURE_PROBLEM_TYPE] = numpy.array(
        inputs[FEATURE_PROBLEM_TYPE], dtype="int8"
    )
    return inputs, numpy.array(outputs)
