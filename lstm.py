import numpy
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    DenseFeatures,
    Embedding,
    Input,
    TimeDistributed,
    add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from mathy.agent.features import FEATURE_BWD_VECTORS  # FEATURE_MOVE_MASK,
from mathy.agent.features import (
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    TRAIN_LABELS_TARGET_VALUE,
    get_max_lengths,
    parse_example_for_training,
)
from mathy.agent.layers.math_policy_dropout import MathPolicyDropout
from mathy.agent.layers.resnet_stack import ResNetStack
from mathy.agent.training.math_experience import MathExamples
from mathy.core.expressions import MathTypeKeysMax

# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()

num_actions = 6


def build_training_data():
    data = MathExamples("overfit.jsonl")
    main_inputs = []
    main_labels = []

    # aux_inputs = []
    # aux_labels = []

    # rpred_inputs = []
    # rpred_labels = []
    max_pi_sequence, max_sequence = get_max_lengths(data.examples)
    for ex in data.examples:
        features, labels = parse_example_for_training(ex, max_sequence, max_pi_sequence)
        # sequence_features = {
        #     FEATURE_BWD_VECTORS: numpy.array(features[FEATURE_BWD_VECTORS]),
        #     FEATURE_FWD_VECTORS: numpy.array(features[FEATURE_FWD_VECTORS]),
        #     FEATURE_LAST_BWD_VECTORS: numpy.array(features[FEATURE_LAST_BWD_VECTORS]),
        #     FEATURE_LAST_FWD_VECTORS: numpy.array(features[FEATURE_LAST_FWD_VECTORS]),
        # }
        # prev_seq = numpy.array(features[FEATURE_LAST_FWD_VECTORS])
        # curr_seq = numpy.array(features[FEATURE_FWD_VECTORS])

        # rpred_inputs.append(prev_seq + curr_seq)
        # rpred_labels.append(ex["discounted"])
        main_inputs.append(features)
        main_labels.append(labels)
        # aux_inputs.append(numpy.array(ex["action"], dtype="float32"))
        # aux_labels.append(numpy.array(ex["reward"], dtype="float32"))

    main_inputs = numpy.array(main_inputs)
    main_labels = numpy.array(main_labels)
    # rpred_inputs = numpy.array(rpred_inputs)
    # rpred_labels = numpy.array(rpred_labels)
    # aux_inputs = numpy.array(aux_inputs)
    # aux_labels = numpy.array(aux_labels)

    examples = {
        "pi_in": main_inputs,
        # "aux_gctrl_in": aux_inputs,
        # "rpred_in": rpred_inputs,
    }
    labels = {
        "pi_out": main_labels,
        # "aux_gctrl_out": aux_labels,
        # "rpred_out": rpred_labels,
    }
    return examples, labels


def build_feature_columns():
    """Build out the Tensorflow Feature Columns that define the inputs from Mathy
    into the neural network. 
    
    Returns tuple of (ctx_columns, seq_columns)"""
    f_move_count = tf.feature_column.numeric_column(
        key=FEATURE_MOVE_COUNTER, dtype=tf.int64
    )
    f_moves_remaining = tf.feature_column.numeric_column(
        key=FEATURE_MOVES_REMAINING, dtype=tf.int64
    )
    f_last_rule = tf.feature_column.numeric_column(
        key=FEATURE_LAST_RULE, dtype=tf.int64
    )
    f_node_count = tf.feature_column.numeric_column(
        key=FEATURE_NODE_COUNT, dtype=tf.int64
    )
    f_problem_type = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity(
            key=FEATURE_PROBLEM_TYPE, num_buckets=32
        )
    )
    feature_columns = [
        f_problem_type,
        f_last_rule,
        f_node_count,
        f_move_count,
        f_moves_remaining,
    ]

    vocab_buckets = MathTypeKeysMax + 1

    #
    # Sequence features
    #
    # feat_policy_mask = tf.feature_column.sequence_numeric_column(
    #     key=FEATURE_MOVE_MASK, dtype=tf.int64, shape=action_size
    # )
    feat_bwd_vectors = tf.feature_column.embedding_column(
        tf.feature_column.sequence_categorical_column_with_identity(
            key=FEATURE_BWD_VECTORS, num_buckets=vocab_buckets
        ),
        dimension=32,
    )
    feat_fwd_vectors = tf.feature_column.embedding_column(
        tf.feature_column.sequence_categorical_column_with_identity(
            key=FEATURE_FWD_VECTORS, num_buckets=vocab_buckets
        ),
        dimension=32,
    )
    feat_last_bwd_vectors = tf.feature_column.embedding_column(
        tf.feature_column.sequence_categorical_column_with_identity(
            key=FEATURE_LAST_BWD_VECTORS, num_buckets=vocab_buckets
        ),
        dimension=32,
    )
    feat_last_fwd_vectors = tf.feature_column.embedding_column(
        tf.feature_column.sequence_categorical_column_with_identity(
            key=FEATURE_LAST_FWD_VECTORS, num_buckets=vocab_buckets
        ),
        dimension=32,
    )
    sequence_columns = [
        feat_fwd_vectors,
        feat_bwd_vectors,
        feat_last_fwd_vectors,
        feat_last_bwd_vectors,
    ]
    return feature_columns, sequence_columns


def build_model():
    shared_units = 32

    # 1 token with 2 extract windows of the tokens on either side.
    # ctx_vector_len = 9
    # # one set of vectors for forward and another for backward
    # bi_len = ctx_vector_len * 2
    # # the number of timesteps the reward prediction model is given
    # rpred_timesteps = 2

    feature_columns, sequence_columns = build_feature_columns()

    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="seq_features"
    )(sequence_features)
    context_inputs = DenseFeatures(feature_columns, name="ctx_features")(features)

    # shared_net = ResNetStack(shared_units, share_weights=True)
    # rpred_in = Input(shape=(ctx_vector_len), dtype="float32", name="rpred_in")
    pi_in = Input(shape=(None, num_actions), dtype="float32", name="pi_in")
    # auxiliary_input = Input(shape=(1,), name="aux_gctrl_in")
    seq_vectors = Embedding(output_dim=128, input_dim=64)(rpred_in)
    lstm_out = LSTM(shared_units, return_sequences=True)(seq_vectors)
    last_out = lstm_out[:, -1]
    last_out.set_shape([None, lstm_out.get_shape()[-1]])
    # auxiliary_output = Dense(1, activation="relu", name="aux_gctrl_out")(
    #     add([last_out, shared_net(auxiliary_input)])
    # )
    # policy_net = TimeDistributed(MathPolicyDropout(num_actions), name="pi_out")
    rpred_out = TimeDistributed(MathPolicyDropout(num_actions), name="rpred_out")(
        lstm_out
    )
    # pi_out = policy_net(lstm_out)
    return Model(inputs=[rpred_in], outputs=[rpred_out])


examples, labels, max_seq_len = build_training_data()
model = build_model(max_seq_len)
model.compile(optimizer="adam", loss={"rpred_out": "mean_squared_error"})
model.summary()
plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.fit(
    examples,
    labels,
    epochs=1000,
    batch_size=2048,
    callbacks=[TensorBoard(log_dir="./training/keras/", write_graph=True)],
)
