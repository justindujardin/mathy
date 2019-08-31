import numpy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    LSTM,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from mathy import (
    FEATURE_FWD_VECTORS,
    MathTypeKeysMax,
    get_max_lengths,
    parse_example_for_training,
)
from mathy.agent.layers.math_policy_dropout import MathPolicyDropout


class MathExamples:
    """Load and save a list of training examples from the file system"""

    _examples: List[MathyEnvObservation]
    file_path: str

    def __init__(self, file_path, initial_load=True):
        self.file_path = file_path
        self._examples = []
        if initial_load:
            self.load()

    @property
    def examples(self) -> List[MathyEnvObservation]:
        """Get the entire list of examples"""
        return self._examples

    def add(self, examples: List[MathyEnvObservation]):
        self._examples += examples

    def load(self):
        # Try to match a specified file first
        file_path = Path(self.file_path)
        if not file_path.is_file():
            raise ValueError(f"file '{file_path}' does not exist!")
        examples: List[MathyEnvObservation] = []
        with file_path.open("r", encoding="utf8") as f:
            for line in f:
                ex = MathyEnvObservation(**ujson.loads(line))
                examples.append(ex)
        self._examples = examples
        return True

    def save(self, to_file=None):
        """Save the accumulated experience to a file. Defaults to the file it
        was loaded from"""
        if to_file is None:
            to_file = self.file_path
        experience_folder = Path(self.file_path).parent
        if not experience_folder.is_dir():
            experience_folder.mkdir(parents=True, exist_ok=True)

        # Write to local file then copy over (don't thrash virtual file systems
        # like GCS)
        fd, tmp_file = tempfile.mkstemp()
        with Path(tmp_file).open("w", encoding="utf-8") as f:
            for line in self._examples:
                f.write(
                    ujson.dumps(line._as_dict(), escape_forward_slashes=False) + "\n"
                )

        out_file = Path(self.file_path)
        if out_file.is_file():
            copyfile(str(out_file), f"{str(out_file)}.bak")
        copyfile(tmp_file, str(out_file))
        os.remove(tmp_file)
        os.close(fd)
        return str(out_file)


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
