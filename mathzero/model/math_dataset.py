from pathlib import Path
import ujson
from ..environment_state import INPUT_EXAMPLES_FILE_NAME
from ..model.features import (
    FEATURE_NODE_COUNT,
    FEATURE_FWD_VECTORS,
    FEATURE_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_PROBLEM_TYPE,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_VALUE,
    TRAIN_LABELS_AS_MATRIX,
    parse_example_for_training,
)


def make_training_input_fn(examples, batch_size):
    """Return an input function that lazily loads self-play examples from 
    the given file during training
    """
    import tensorflow as tf

    output_types = (
        {
            FEATURE_FWD_VECTORS: tf.int8,
            FEATURE_BWD_VECTORS: tf.int8,
            FEATURE_LAST_FWD_VECTORS: tf.int8,
            FEATURE_LAST_BWD_VECTORS: tf.int8,
            FEATURE_NODE_COUNT: tf.int32,
            FEATURE_MOVE_COUNTER: tf.int32,
            FEATURE_MOVES_REMAINING: tf.int32,
            FEATURE_PROBLEM_TYPE: tf.int32,
        },
        {
            TRAIN_LABELS_TARGET_PI: tf.float32,
            TRAIN_LABELS_TARGET_VALUE: tf.float32,
        },
    )

    lengths = [len(l["inputs"][FEATURE_BWD_VECTORS]) for l in examples]

    max_sequence = max(lengths)

    def _lazy_examples():
        nonlocal max_sequence
        for ex in examples:
            yield parse_example_for_training(ex, max_sequence)

    def _input_fn():
        nonlocal output_types

        dataset = tf.data.Dataset.from_generator(
            _lazy_examples, output_types=output_types
        )
        # Shuffled from long-term memory  
        # dataset = dataset.shuffle(50000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size)
        return dataset

    return _input_fn
