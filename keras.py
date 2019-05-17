import numpy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from mathy.agent.features import get_max_lengths, pad_array
from mathy.agent.layers.math_policy_dropout import MathPolicyDropout
from mathy.agent.layers.resnet_stack import ResNetStack
from mathy.agent.training.math_experience import MathExamples

# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()

num_actions = 6


def build_training_data():
    data = MathExamples("overfit.jsonl")
    main_inputs = []
    main_labels = []
    aux_inputs = []
    aux_labels = []
    seq_pad_value = tuple([0.0] * num_actions)
    max_pi_sequence, max_sequence = get_max_lengths(data.examples)
    for ex in data.examples:
        assert len(ex["policy"]) == len(ex["policy_mask"])
        pi = pad_array(ex["policy"], max_pi_sequence, value=seq_pad_value)
        pi_mask = pad_array(ex["policy_mask"], max_pi_sequence, value=seq_pad_value)
        main_inputs.append(numpy.array(pi, dtype="float32"))
        main_labels.append(numpy.array(pi_mask, dtype="float32"))
        aux_inputs.append(numpy.array(ex["action"], dtype="float32"))
        aux_labels.append(numpy.array(ex["reward"], dtype="float32"))
    main_inputs = numpy.array(main_inputs, dtype="float32")
    main_labels = numpy.array(main_labels, dtype="float32")
    aux_inputs = numpy.array(aux_inputs, dtype="float32")
    aux_labels = numpy.array(aux_labels, dtype="float32")

    examples = {"pi_in": main_inputs, "aux_gctrl_in": aux_inputs}
    labels = {"pi_out": main_labels, "aux_gctrl_out": aux_labels}
    return examples, labels


def build_model():
    shared_units = 32
    shared_net = ResNetStack(shared_units, share_weights=True)
    pi_in = Input(shape=(None, num_actions), dtype="float32", name="pi_in")
    auxiliary_input = Input(shape=(1,), name="aux_gctrl_in")
    lstm_out = LSTM(shared_units, return_sequences=True)(pi_in)
    last_out = lstm_out[:, -1]
    last_out.set_shape([None, lstm_out.get_shape()[-1]])
    auxiliary_output = Dense(1, activation="relu", name="aux_gctrl_out")(
        add([last_out, shared_net(auxiliary_input)])
    )
    policy_net = TimeDistributed(MathPolicyDropout(num_actions), name="pi_out")
    pi_out = policy_net(lstm_out)
    return Model(inputs=[pi_in, auxiliary_input], outputs=[pi_out, auxiliary_output])


model = build_model()
examples, labels = build_training_data()
model.compile(
    optimizer="adam",
    loss={"pi_out": "mean_squared_error", "aux_gctrl_out": "mean_squared_error"},
)
model.summary()
plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.fit(
    examples,
    labels,
    epochs=1000,
    batch_size=2048,
    callbacks=[TensorBoard(log_dir="./training/keras/", write_graph=True)],
)
