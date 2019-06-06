import numpy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    LSTM,
    Embedding,
    Input,
    TimeDistributed,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from mathy.core.expressions import MathTypeKeysMax
from mathy.agent.features import (
    FEATURE_FWD_VECTORS,
    get_max_lengths,
    parse_example_for_training,
)
from mathy.agent.layers.math_policy_dropout import MathPolicyDropout
from mathy.agent.training.math_experience import MathExamples

num_actions = 6


def build_training_data():
    data = MathExamples("overfit.jsonl")
    main_inputs = []
    main_labels = []
    max_pi_sequence, max_sequence = get_max_lengths(data.examples)
    for ex in data.examples:
        features, labels = parse_example_for_training(ex, max_sequence, max_pi_sequence)
        curr_seq = numpy.array(features[FEATURE_FWD_VECTORS])
        main_inputs.append(curr_seq)
        main_labels.append(labels["policy"])
    main_inputs = numpy.array(main_inputs)
    main_labels = numpy.array(main_labels)
    examples = {"pi_in": main_inputs}
    labels = {"pi_out": main_labels}
    return examples, labels


def build_model():
    shared_units = 32
    # 1 token with 2 extract windows of the tokens on either side.
    ctx_vector_len = 9
    embedding_dim = 24
    pi_in = Input(
        batch_shape=(None, None, ctx_vector_len), dtype="float32", name="pi_in"
    )
    embedding = Embedding(MathTypeKeysMax, embedding_dim, input_length=3)
    embed = embedding(pi_in)
    lstm_out = LSTM(shared_units, return_sequences=True)(embed)
    last_out = lstm_out[:, -1]
    last_out.set_shape([None, lstm_out.get_shape()[-1]])
    policy_net = TimeDistributed(MathPolicyDropout(num_actions), name="pi_out")
    pi_out = policy_net(lstm_out)
    return Model(inputs=[pi_in], outputs=[pi_out])


examples, labels = build_training_data()
model = build_model()
model.compile(optimizer="adam", loss={"pi_out": "mean_squared_error"})
model.summary()
plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.fit(
    examples,
    labels,
    epochs=1000,
    batch_size=2048,
    callbacks=[TensorBoard(log_dir="./training/keras/", write_graph=True)],
)
