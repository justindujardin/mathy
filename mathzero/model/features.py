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


def pad_array(A, max_length, value=0):
    """Pad a numpy array to the given size with the given padding value"""
    a_len = len(A)
    if a_len >= max_length:
        return A
    t = max_length - len(A)
    return numpy.pad(A, pad_width=(0, t), mode="constant", constant_values=value)


TRAIN_LABELS_TARGET_PI = "policy"
TRAIN_LABELS_TARGET_REWARD = "value"
TRAIN_LABELS_TARGET_FOCUS = "focus"
TRAIN_LABELS_AS_MATRIX = "matrix"


def parse_examples_for_training(examples, unwrap_single=True):
    """Parse a given JSONL dataset of examples into an x/y output
    params:
        examples: the JSONL items from `examples.jsonl`
    returns: 
        tuple of (examples, labels) for training where labels are a tuple 
        of (target_policy, target_reward, target_focus)
    """
    import tensorflow as tf

    inputs = {}
    outputs = {
        TRAIN_LABELS_TARGET_PI: [],
        TRAIN_LABELS_TARGET_REWARD: [],
        TRAIN_LABELS_TARGET_FOCUS: [],
        TRAIN_LABELS_AS_MATRIX: [],
    }
    for feature_key in FEATURE_COLUMNS:
        inputs[feature_key] = []
    # Build up a feature map that can work as input
    example_len = len(examples)
    for ex in examples:
        ex_input = ex["inputs"]
        ex_append = {}
        for feature_key in FEATURE_COLUMNS:
            inputs[feature_key].append(ex_input[feature_key])
        target_pi = numpy.array(ex["policy"], dtype="float32")
        target_reward = ex["reward"]
        target_focus = ex["focus"]
        outputs[TRAIN_LABELS_TARGET_PI].append(target_pi)
        outputs[TRAIN_LABELS_TARGET_REWARD].append(target_reward)
        outputs[TRAIN_LABELS_TARGET_FOCUS].append(target_focus)
        outputs[TRAIN_LABELS_AS_MATRIX].append([target_pi, target_reward, target_focus])

    # Pad the variable length columns to longest in the list
    inputs[FEATURE_TOKEN_TYPES] = tf.contrib.layers.dense_to_sparse(
        features[FEATURE_TOKEN_TYPES], eos_token=-1
    )
    inputs[FEATURE_TOKEN_VALUES] = tf.contrib.layers.dense_to_sparse(
        features[FEATURE_TOKEN_VALUES], eos_token=""
    )
    # inputs[FEATURE_TOKEN_TYPES] = seq_tensor
    # inputs[FEATURE_TOKEN_TYPES] = numpy.array(
    #     list(zip_longest(*inputs[FEATURE_TOKEN_TYPES], fillvalue=0))
    # ).T
    # inputs[FEATURE_TOKEN_VALUES] = numpy.array(
    #     list(zip_longest(*inputs[FEATURE_TOKEN_VALUES], fillvalue=0))
    # ).T

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
    outputs[TRAIN_LABELS_TARGET_PI] = numpy.array(outputs[TRAIN_LABELS_TARGET_PI])
    outputs[TRAIN_LABELS_TARGET_REWARD] = numpy.array(
        outputs[TRAIN_LABELS_TARGET_REWARD]
    )
    outputs[TRAIN_LABELS_TARGET_FOCUS] = numpy.array(outputs[TRAIN_LABELS_TARGET_FOCUS])
    return inputs, outputs


def parse_example_for_training(example, max_sequence=None):
    """Wrapper that accepts a single example to parse for feeding to the network.
    
    Returns: a tuple of(features, labels) """
    inputs = {}
    outputs = {
        TRAIN_LABELS_TARGET_PI: [],
        TRAIN_LABELS_TARGET_REWARD: [],
        TRAIN_LABELS_TARGET_FOCUS: [],
    }
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
    outputs[TRAIN_LABELS_TARGET_PI] = example["policy"]
    outputs[TRAIN_LABELS_TARGET_REWARD] = [example["reward"]]
    outputs[TRAIN_LABELS_TARGET_FOCUS] = [example["focus"]]
    # outputs[TRAIN_LABELS_AS_MATRIX] = [
    #     example["policy"],
    #     example["reward"],
    #     example["focus"],
    # ]
    inputs[FEATURE_NODE_COUNT] = len(ex_input[FEATURE_TOKEN_TYPES])
    return inputs, outputs
