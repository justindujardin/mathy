import numpy
import tensorflow as tf

from ..agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_MOVE_MASK,
    FEATURE_PROBLEM_TYPE,
    TENSOR_KEY_NODE_CTRL,
    TENSOR_KEY_GROUPING_CTRL,
    TENSOR_KEY_GROUP_PREDICT,
    TENSOR_KEY_REWARD_PREDICT,
    TENSOR_KEY_PI,
    TENSOR_KEY_VALUE,
    parse_example_for_training,
)


def make_training_input_fn(examples, batch_size):
    """Return an input function that lazily loads self-play examples from
    the given file during training."""

    output_types = (
        {
            FEATURE_FWD_VECTORS: tf.int64,
            FEATURE_BWD_VECTORS: tf.int64,
            FEATURE_LAST_FWD_VECTORS: tf.int64,
            FEATURE_LAST_BWD_VECTORS: tf.int64,
            FEATURE_LAST_RULE: tf.int64,
            FEATURE_NODE_COUNT: tf.int64,
            FEATURE_MOVE_COUNTER: tf.int64,
            FEATURE_MOVES_REMAINING: tf.int64,
            FEATURE_PROBLEM_TYPE: tf.int64,
            FEATURE_MOVE_MASK: tf.int64,
        },
        {
            TENSOR_KEY_PI: tf.float32,
            TENSOR_KEY_NODE_CTRL: tf.int32,
            TENSOR_KEY_GROUPING_CTRL: tf.int32,
            TENSOR_KEY_GROUP_PREDICT: tf.int32,
            TENSOR_KEY_REWARD_PREDICT: tf.int32,
            TENSOR_KEY_VALUE: tf.float32,
        },
    )

    lengths = [len(l["features"][FEATURE_BWD_VECTORS]) for l in examples]
    pi_lengths = [len(numpy.array(l[TENSOR_KEY_PI]).flatten()) for l in examples]

    max_sequence = max(lengths)
    max_pi_sequence = max(pi_lengths)

    def _lazy_examples():
        nonlocal max_sequence
        for ex in examples:
            yield parse_example_for_training(ex, max_sequence, max_pi_sequence)

    def _input_fn():
        nonlocal output_types

        dataset = tf.data.Dataset.from_generator(
            _lazy_examples, output_types=output_types
        )
        # Shuffled during long-term memory extraction
        # dataset = dataset.shuffle(50000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size)
        return dataset

    return _input_fn
